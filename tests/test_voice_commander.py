"""Unit and Integration Tests for Voice-to-Intent Pipeline (Task 2).

Tests:
1. test_voice_intent_enum: 10 intent enum members.
2. test_parse_text_vietnamese: Keyword spotting for Vietnamese phrases.
3. test_parse_text_english: Keyword spotting for English phrases.
4. test_dispatch_intent_all_intents: Mapping intents to cmd_mode, cmd_vel, EmotionState.
5. test_uart_packet_packing_and_unpacking: Binary UART packet packing, unpacking, CRC16 validation.
6. test_apply_to_runner_integration: Updating ParallelSimRunner state (all envs and single env).
"""

from pathlib import Path
import sys
import numpy as np
import pytest

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.env_config import EmotionState, LocomotionMode
from training.run_parallel_sim import ParallelSimRunner
from training.voice_commander import (
    VoiceCommander,
    VoiceIntent,
    calc_crc16,
    pack_uart_voice_packet,
    unpack_uart_voice_packet,
)


def test_voice_intent_enum():
    """Verify VoiceIntent enum contains all 10 required intent definitions."""
    expected_members = [
        "SIT",
        "LIE",
        "STAND_WALK",
        "STOP",
        "HAPPY",
        "PROUD",
        "SAD",
        "CURIOUS",
        "TURN_LEFT",
        "TURN_RIGHT",
    ]
    assert len(VoiceIntent) == 10
    for member in expected_members:
        assert hasattr(VoiceIntent, member)
        intent = getattr(VoiceIntent, member)
        assert isinstance(intent, VoiceIntent)


def test_parse_text_vietnamese():
    """Verify Vietnamese speech recognition keywords mapping to intents."""
    vc = VoiceCommander()

    vietnamese_cases = {
        "ngồi xuống": VoiceIntent.SIT,
        "ngồi": VoiceIntent.SIT,
        "nằm xuống": VoiceIntent.LIE,
        "nằm": VoiceIntent.LIE,
        "đứng dậy": VoiceIntent.STAND_WALK,
        "bước đi": VoiceIntent.STAND_WALK,
        "dừng lại": VoiceIntent.STOP,
        "vui lên": VoiceIntent.HAPPY,
        "tự hào": VoiceIntent.PROUD,
        "buồn bã": VoiceIntent.SAD,
        "tò mò": VoiceIntent.CURIOUS,
        "quay trái": VoiceIntent.TURN_LEFT,
        "quay phải": VoiceIntent.TURN_RIGHT,
    }

    for phrase, expected_intent in vietnamese_cases.items():
        assert vc.parse_text(phrase) == expected_intent, f"Failed for '{phrase}'"
        # Test case insensitivity and punctuation tolerance
        assert vc.parse_text(f"  {phrase.upper()}!  ") == expected_intent
        assert VoiceCommander.parse_text(f"duck {phrase} nào") == expected_intent


def test_parse_text_english():
    """Verify English speech recognition keywords mapping to intents and noise handling."""
    english_cases = {
        "sit": VoiceIntent.SIT,
        "lie down": VoiceIntent.LIE,
        "sleep": VoiceIntent.LIE,
        "stand up": VoiceIntent.STAND_WALK,
        "walk": VoiceIntent.STAND_WALK,
        "stop": VoiceIntent.STOP,
        "happy": VoiceIntent.HAPPY,
        "proud": VoiceIntent.PROUD,
        "sad": VoiceIntent.SAD,
        "curious": VoiceIntent.CURIOUS,
        "turn left": VoiceIntent.TURN_LEFT,
        "turn right": VoiceIntent.TURN_RIGHT,
    }

    for phrase, expected_intent in english_cases.items():
        assert VoiceCommander.parse_text(phrase) == expected_intent, f"Failed for '{phrase}'"
        assert VoiceCommander.parse_text(f"PLEASE {phrase.upper()} NOW.") == expected_intent

    # Unrecognized command returns None
    assert VoiceCommander.parse_text("some random noise 1234") is None
    assert VoiceCommander.parse_text("") is None


def test_dispatch_intent_all_intents():
    """Verify dispatch_intent returns correct cmd_mode, cmd_vel, and EmotionState for all 10 intents."""
    vc = VoiceCommander()

    # 1. SIT
    d_sit = vc.dispatch_intent(VoiceIntent.SIT)
    assert d_sit["cmd_mode"] == LocomotionMode.SIT
    assert np.allclose(d_sit["cmd_vel"], [0.0, 0.0, 0.0])
    assert d_sit["emotion"] == EmotionState.NEUTRAL

    # 2. LIE
    d_lie = vc.dispatch_intent(VoiceIntent.LIE)
    assert d_lie["cmd_mode"] == LocomotionMode.LIE
    assert np.allclose(d_lie["cmd_vel"], [0.0, 0.0, 0.0])
    assert d_lie["emotion"] == EmotionState.SLEEPY

    # 3. STAND_WALK
    d_walk = vc.dispatch_intent(VoiceIntent.STAND_WALK)
    assert d_walk["cmd_mode"] == LocomotionMode.WALK
    assert np.allclose(d_walk["cmd_vel"], [0.5, 0.0, 0.0])
    assert d_walk["emotion"] == EmotionState.NEUTRAL

    # 4. STOP
    d_stop = vc.dispatch_intent(VoiceIntent.STOP)
    assert d_stop["cmd_mode"] == LocomotionMode.WALK
    assert np.allclose(d_stop["cmd_vel"], [0.0, 0.0, 0.0])
    assert d_stop["emotion"] == EmotionState.NEUTRAL

    # 5. HAPPY
    d_happy = vc.dispatch_intent(VoiceIntent.HAPPY)
    assert d_happy["cmd_mode"] == LocomotionMode.WALK
    assert np.allclose(d_happy["cmd_vel"], [0.0, 0.0, 0.0])
    assert d_happy["emotion"] == EmotionState.HAPPY

    # 6. PROUD
    d_proud = vc.dispatch_intent(VoiceIntent.PROUD)
    assert d_proud["cmd_mode"] == LocomotionMode.WALK
    assert np.allclose(d_proud["cmd_vel"], [0.0, 0.0, 0.0])
    assert d_proud["emotion"] == EmotionState.PROUD

    # 7. SAD
    d_sad = vc.dispatch_intent(VoiceIntent.SAD)
    assert d_sad["cmd_mode"] == LocomotionMode.WALK
    assert np.allclose(d_sad["cmd_vel"], [0.0, 0.0, 0.0])
    assert d_sad["emotion"] == EmotionState.SAD

    # 8. CURIOUS
    d_curious = vc.dispatch_intent(VoiceIntent.CURIOUS)
    assert d_curious["cmd_mode"] == LocomotionMode.WALK
    assert np.allclose(d_curious["cmd_vel"], [0.0, 0.0, 0.0])
    assert d_curious["emotion"] == EmotionState.CURIOUS

    # 9. TURN_LEFT
    d_tl = vc.dispatch_intent(VoiceIntent.TURN_LEFT)
    assert d_tl["cmd_mode"] == LocomotionMode.WALK
    assert d_tl["cmd_vel"][2] > 0.0
    assert d_tl["emotion"] == EmotionState.NEUTRAL

    # 10. TURN_RIGHT
    d_tr = vc.dispatch_intent(VoiceIntent.TURN_RIGHT)
    assert d_tr["cmd_mode"] == LocomotionMode.WALK
    assert d_tr["cmd_vel"][2] < 0.0
    assert d_tr["emotion"] == EmotionState.NEUTRAL


def test_uart_packet_packing_and_unpacking():
    """Verify binary UART packet format with sync bytes, payload length, and CRC16 checksum."""
    seq = 42
    raw = pack_uart_voice_packet(VoiceIntent.LIE, seq_id=seq)

    # Frame Structure: [0xAA] [0x55] [0x06] [seq] [len_l] [len_h] [14 bytes payload] [crc_l] [crc_h]
    # Total: 6 (header) + 14 (payload) + 2 (crc) = 22 bytes
    assert len(raw) == 22
    assert raw[0] == 0xAA
    assert raw[1] == 0x55
    assert raw[2] == 0x06  # PKT_TYPE_VOICE_INTENT_CMD
    assert raw[3] == seq
    payload_len = raw[4] | (raw[5] << 8)
    assert payload_len == 14

    # Verify CRC16
    computed_crc = calc_crc16(raw[:20])
    rx_crc = raw[20] | (raw[21] << 8)
    assert computed_crc == rx_crc

    # Decode packet
    unpacked = unpack_uart_voice_packet(raw)
    assert unpacked["packet_type"] == 0x06
    assert unpacked["seq_id"] == seq
    assert unpacked["cmd_mode"] == LocomotionMode.LIE
    assert unpacked["emotion"] == EmotionState.SLEEPY
    assert np.allclose(unpacked["cmd_vel"], [0.0, 0.0, 0.0])

    # Error handling 1: CRC corruption
    corrupted_crc = bytearray(raw)
    corrupted_crc[20] ^= 0xFF
    with pytest.raises(ValueError, match="CRC mismatch"):
        unpack_uart_voice_packet(bytes(corrupted_crc))

    # Error handling 2: Invalid sync bytes
    corrupted_sync = bytearray(raw)
    corrupted_sync[0] = 0x00
    with pytest.raises(ValueError, match="Invalid sync"):
        unpack_uart_voice_packet(bytes(corrupted_sync))

    # Error handling 3: Truncated frame
    with pytest.raises(ValueError, match="Incomplete packet"):
        unpack_uart_voice_packet(raw[:10])


def test_apply_to_runner_integration():
    """Verify applying voice intent updates cmd_modes, commands, and current_emotions in ParallelSimRunner."""
    runner = ParallelSimRunner(num_envs=4, stand_still=False)

    # Broadcast VoiceIntent.SIT to all environments
    VoiceCommander.apply_to_runner(runner, VoiceIntent.SIT)
    assert np.all(runner.cmd_modes == int(LocomotionMode.SIT))
    assert np.allclose(runner.commands, 0.0)
    assert np.all(runner.current_emotions == int(EmotionState.NEUTRAL))

    # Broadcast VoiceIntent.LIE to all environments
    VoiceCommander.apply_to_runner(runner, VoiceIntent.LIE)
    assert np.all(runner.cmd_modes == int(LocomotionMode.LIE))
    assert np.allclose(runner.commands, 0.0)
    assert np.all(runner.current_emotions == int(EmotionState.SLEEPY))

    # Broadcast VoiceIntent.STAND_WALK
    VoiceCommander.apply_to_runner(runner, VoiceIntent.STAND_WALK)
    assert np.all(runner.cmd_modes == int(LocomotionMode.WALK))
    assert np.allclose(runner.commands[:, 0], 0.5)

    # Targeted apply to single environment env_idx=1
    VoiceCommander.apply_to_runner(runner, VoiceIntent.HAPPY, env_idx=1)
    assert runner.current_emotions[1] == int(EmotionState.HAPPY)
    # Other environments should remain unchanged
    assert runner.current_emotions[0] == int(EmotionState.NEUTRAL)
    assert runner.current_emotions[2] == int(EmotionState.NEUTRAL)
    assert runner.current_emotions[3] == int(EmotionState.NEUTRAL)
