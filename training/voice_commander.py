"""Bilingual Voice-to-Intent Pipeline & Hardware UART Protocol for MicroDuck 8-DOF.

Provides:
- VoiceIntent enum (10 locomotion and expressive intents)
- Keyword Spotting for Vietnamese and English speech commands
- Intent dispatcher mapping to LocomotionMode, cmd_vel, and EmotionState
- ParallelSimRunner integration for simulation testing
- Binary UART packet pack/unpack with CRC16 (PKT_TYPE_VOICE_INTENT_CMD = 0x06)
"""

from enum import Enum
import re
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from training.env_config import EmotionState, LocomotionMode


class VoiceIntent(str, Enum):
    """10 voice intents for locomotion commands and expressive head postures."""

    SIT = "SIT"
    LIE = "LIE"
    STAND_WALK = "STAND_WALK"
    STOP = "STOP"
    HAPPY = "HAPPY"
    PROUD = "PROUD"
    SAD = "SAD"
    CURIOUS = "CURIOUS"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"


# Lookup table for bilingual keyword spotting (phrase, VoiceIntent)
# Ordered by phrase length descending to match composite multi-word keywords first
KEYWORD_MAPPINGS: List[Tuple[str, VoiceIntent]] = [
    # Vietnamese
    ("ngồi xuống", VoiceIntent.SIT),
    ("nằm xuống", VoiceIntent.LIE),
    ("đứng dậy", VoiceIntent.STAND_WALK),
    ("bước đi", VoiceIntent.STAND_WALK),
    ("dừng lại", VoiceIntent.STOP),
    ("vui lên", VoiceIntent.HAPPY),
    ("tự hào", VoiceIntent.PROUD),
    ("buồn bã", VoiceIntent.SAD),
    ("tò mò", VoiceIntent.CURIOUS),
    ("quay trái", VoiceIntent.TURN_LEFT),
    ("quay phải", VoiceIntent.TURN_RIGHT),
    ("ngồi", VoiceIntent.SIT),
    ("nằm", VoiceIntent.LIE),
    # English
    ("lie down", VoiceIntent.LIE),
    ("stand up", VoiceIntent.STAND_WALK),
    ("turn left", VoiceIntent.TURN_LEFT),
    ("turn right", VoiceIntent.TURN_RIGHT),
    ("sit", VoiceIntent.SIT),
    ("sleep", VoiceIntent.LIE),
    ("walk", VoiceIntent.STAND_WALK),
    ("stop", VoiceIntent.STOP),
    ("happy", VoiceIntent.HAPPY),
    ("proud", VoiceIntent.PROUD),
    ("sad", VoiceIntent.SAD),
    ("curious", VoiceIntent.CURIOUS),
]

# Dispatch configuration for each intent
INTENT_DISPATCH_MAP: Dict[VoiceIntent, Dict[str, Any]] = {
    VoiceIntent.SIT: {
        "cmd_mode": LocomotionMode.SIT,
        "cmd_vel": [0.0, 0.0, 0.0],
        "emotion": EmotionState.NEUTRAL,
    },
    VoiceIntent.LIE: {
        "cmd_mode": LocomotionMode.LIE,
        "cmd_vel": [0.0, 0.0, 0.0],
        "emotion": EmotionState.SLEEPY,
    },
    VoiceIntent.STAND_WALK: {
        "cmd_mode": LocomotionMode.WALK,
        "cmd_vel": [0.5, 0.0, 0.0],
        "emotion": EmotionState.NEUTRAL,
    },
    VoiceIntent.STOP: {
        "cmd_mode": LocomotionMode.WALK,
        "cmd_vel": [0.0, 0.0, 0.0],
        "emotion": EmotionState.NEUTRAL,
    },
    VoiceIntent.HAPPY: {
        "cmd_mode": LocomotionMode.WALK,
        "cmd_vel": [0.0, 0.0, 0.0],
        "emotion": EmotionState.HAPPY,
    },
    VoiceIntent.PROUD: {
        "cmd_mode": LocomotionMode.WALK,
        "cmd_vel": [0.0, 0.0, 0.0],
        "emotion": EmotionState.PROUD,
    },
    VoiceIntent.SAD: {
        "cmd_mode": LocomotionMode.WALK,
        "cmd_vel": [0.0, 0.0, 0.0],
        "emotion": EmotionState.SAD,
    },
    VoiceIntent.CURIOUS: {
        "cmd_mode": LocomotionMode.WALK,
        "cmd_vel": [0.0, 0.0, 0.0],
        "emotion": EmotionState.CURIOUS,
    },
    VoiceIntent.TURN_LEFT: {
        "cmd_mode": LocomotionMode.WALK,
        "cmd_vel": [0.0, 0.0, 0.5],
        "emotion": EmotionState.NEUTRAL,
    },
    VoiceIntent.TURN_RIGHT: {
        "cmd_mode": LocomotionMode.WALK,
        "cmd_vel": [0.0, 0.0, -0.5],
        "emotion": EmotionState.NEUTRAL,
    },
}

# Binary protocol constants
PACKET_SYNC_0 = 0xAA
PACKET_SYNC_1 = 0x55
PKT_TYPE_VOICE_INTENT_CMD = 0x06
VOICE_INTENT_PAYLOAD_LEN = 14  # uint8(1) + uint8(1) + float32(4)*3 = 14 bytes


def calc_crc16(data: bytes) -> int:
    """Calculate CCITT-16 CRC checksum (Polynomial 0x1021, Initial 0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


class VoiceCommander:
    """Bilingual voice command parser, dispatcher, and UART packet encoder/decoder."""

    @classmethod
    def parse_text(cls, text: str) -> Optional[VoiceIntent]:
        """Parse raw speech transcription into VoiceIntent via keyword spotting."""
        if not text or not isinstance(text, str):
            return None

        # Clean punctuation and normalize spacing and lowercase
        cleaned = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).lower()
        padded = f" {' '.join(cleaned.split())} "

        for phrase, intent in KEYWORD_MAPPINGS:
            phrase_clean = f" {phrase.lower()} "
            if phrase_clean in padded:
                return intent

        return None

    @classmethod
    def dispatch_intent(cls, intent: VoiceIntent) -> Dict[str, Any]:
        """Dispatch VoiceIntent to locomotion mode, target velocity, and emotion."""
        if intent not in INTENT_DISPATCH_MAP:
            raise ValueError(f"Unknown VoiceIntent: {intent}")
        return dict(INTENT_DISPATCH_MAP[intent])

    @classmethod
    def apply_to_runner(
        cls,
        runner: Any,
        intent: VoiceIntent,
        env_idx: Optional[int] = None,
    ) -> None:
        """Apply dispatched voice intent targets directly to a ParallelSimRunner."""
        dispatched = cls.dispatch_intent(intent)
        mode = int(dispatched["cmd_mode"])
        vel = np.asarray(dispatched["cmd_vel"], dtype=np.float32)
        emotion = int(dispatched["emotion"])

        if env_idx is None:
            runner.cmd_modes[:] = mode
            runner.commands[:, :] = vel
            runner.current_emotions[:] = emotion
        else:
            runner.cmd_modes[env_idx] = mode
            runner.commands[env_idx, :] = vel
            runner.current_emotions[env_idx] = emotion

    @classmethod
    def pack_uart_voice_packet(cls, intent: VoiceIntent, seq_id: int = 0) -> bytes:
        """Pack voice intent into binary UART frame with CRC16 checksum."""
        dispatched = cls.dispatch_intent(intent)
        mode = int(dispatched["cmd_mode"])
        emotion = int(dispatched["emotion"])
        vx, vy, wz = [float(v) for v in dispatched["cmd_vel"]]

        payload = struct.pack("<BBfff", mode, emotion, vx, vy, wz)
        header = struct.pack(
            "<BBBBH",
            PACKET_SYNC_0,
            PACKET_SYNC_1,
            PKT_TYPE_VOICE_INTENT_CMD,
            seq_id & 0xFF,
            len(payload),
        )
        frame_body = header + payload
        crc = calc_crc16(frame_body)
        return frame_body + struct.pack("<H", crc)

    @classmethod
    def unpack_uart_voice_packet(cls, raw_bytes: bytes) -> Dict[str, Any]:
        """Decode binary UART frame, verifying header sync, payload length, and CRC16."""
        if len(raw_bytes) < 8:
            raise ValueError(f"Incomplete packet: expected at least 8 bytes, got {len(raw_bytes)}")
        if raw_bytes[0] != PACKET_SYNC_0 or raw_bytes[1] != PACKET_SYNC_1:
            raise ValueError(
                f"Invalid sync bytes: 0x{raw_bytes[0]:02X} 0x{raw_bytes[1]:02X}"
            )

        pkt_type = raw_bytes[2]
        seq_id = raw_bytes[3]
        payload_len = struct.unpack("<H", raw_bytes[4:6])[0]

        expected_total = 6 + payload_len + 2
        if len(raw_bytes) < expected_total:
            raise ValueError(
                f"Incomplete packet: expected {expected_total} bytes, got {len(raw_bytes)}"
            )

        computed_crc = calc_crc16(raw_bytes[: 6 + payload_len])
        rx_crc = struct.unpack("<H", raw_bytes[6 + payload_len : 6 + payload_len + 2])[0]
        if computed_crc != rx_crc:
            raise ValueError(
                f"CRC mismatch: computed 0x{computed_crc:04X}, received 0x{rx_crc:04X}"
            )

        payload = raw_bytes[6 : 6 + payload_len]
        if len(payload) != VOICE_INTENT_PAYLOAD_LEN:
            raise ValueError(
                f"Invalid payload length: expected {VOICE_INTENT_PAYLOAD_LEN}, got {len(payload)}"
            )

        mode, emotion, vx, vy, wz = struct.unpack("<BBfff", payload)
        return {
            "packet_type": pkt_type,
            "seq_id": seq_id,
            "cmd_mode": LocomotionMode(mode),
            "emotion": EmotionState(emotion),
            "cmd_vel": [vx, vy, wz],
            "payload_len": payload_len,
        }


# Convenience module-level aliases
parse_text = VoiceCommander.parse_text
dispatch_intent = VoiceCommander.dispatch_intent
apply_to_runner = VoiceCommander.apply_to_runner
pack_uart_voice_packet = VoiceCommander.pack_uart_voice_packet
unpack_uart_voice_packet = VoiceCommander.unpack_uart_voice_packet
