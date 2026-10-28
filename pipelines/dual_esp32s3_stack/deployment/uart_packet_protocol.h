/**
 * @file uart_packet_protocol.h
 * @brief High-Speed Binary Packet Protocol between MCU 1 (Vision/Audio/LiDAR)
 *        and MCU 2 (Locomotion RL Policy) on ESP32-S3 (MicroDuck 8-DOF).
 *
 * Baud Rate: 921,600 bps
 * Frame Structure:
 *   [0xAA] [0x55] [Type] [SeqID] [Len_L] [Len_H] [Payload...] [CRC_L] [CRC_H]
 * Total Overhead: 8 bytes (Header 6 bytes + CRC16 2 bytes)
 */

#ifndef UART_PACKET_PROTOCOL_H
#define UART_PACKET_PROTOCOL_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Sync Bytes
#define PACKET_SYNC_0 0xAA
#define PACKET_SYNC_1 0x55
#define PACKET_HEADER_SIZE 6
#define PACKET_CRC_SIZE 2
#define PACKET_MIN_SIZE (PACKET_HEADER_SIZE + PACKET_CRC_SIZE)
#define PACKET_MAX_PAYLOAD 128

// Packet Types
typedef enum {
    PKT_TYPE_HEARTBEAT         = 0x01,
    PKT_TYPE_VELOCITY_CMD       = 0x02,
    PKT_TYPE_LIDAR_PERCEPTION   = 0x03,
    PKT_TYPE_IMU_DATA          = 0x04,
    PKT_TYPE_JOINT_TELEMETRY   = 0x05,
    PKT_TYPE_EMERGENCY_STOP    = 0xFF
} PacketType_t;

#pragma pack(push, 1)

/**
 * @brief Header structure
 */
typedef struct {
    uint8_t sync0;
    uint8_t sync1;
    uint8_t packet_type;
    uint8_t seq_id;
    uint16_t payload_len;
} PacketHeader;

/**
 * @brief Velocity Command Packet (MCU 1 -> MCU 2)
 */
typedef struct {
    float vx;  // Forward/backward target velocity (m/s)
    float vy;  // Lateral target velocity (m/s)
    float wz;  // Yaw angular target velocity (rad/s)
} VelocityCmdPacket;

/**
 * @brief 5-Ray ToF LiDAR Perception Packet (MCU 1 -> MCU 2)
 */
typedef struct {
    float ranges[5];           // Distances [-30, -15, 0, +15, +30 deg] in meters (0.05 - 2.0m)
    uint8_t obstacle_detected; // Flag: 1 if any ray < 0.35m threshold, 0 otherwise
} LidarPerceptionPacket;

/**
 * @brief Heartbeat Packet (Bidirectional)
 */
typedef struct {
    uint32_t uptime_ms; // Uptime in milliseconds
    uint8_t status;     // 0: OK, 1: WARNING, 2: ERROR
} HeartbeatPacket;

/**
 * @brief Raw IMU Telemetry Packet (MCU 2 -> MCU 1)
 */
typedef struct {
    float gyro[3];  // wx, wy, wz (rad/s)
    float accel[3]; // ax, ay, az (m/s^2)
} ImuDataPacket;

#pragma pack(pop)

/**
 * @brief Calculate CCITT-16 CRC checksum (Polynomial 0x1021, Initial 0xFFFF).
 */
static inline uint16_t calc_crc16(const uint8_t *data, uint16_t len) {
    uint16_t crc = 0xFFFF;
    for (uint16_t i = 0; i < len; ++i) {
        crc ^= ((uint16_t)data[i]) << 8;
        for (uint8_t bit = 0; bit < 8; ++bit) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc = crc << 1;
            }
        }
    }
    return crc;
}

/**
 * @brief Generic function to encode any packet into a wire buffer.
 */
static inline uint16_t encode_packet(
    uint8_t packet_type,
    uint8_t seq_id,
    const void *payload,
    uint16_t payload_len,
    uint8_t *out_buffer,
    uint16_t max_len
) {
    uint16_t total_len = PACKET_HEADER_SIZE + payload_len + PACKET_CRC_SIZE;
    if (total_len > max_len) {
        return 0;
    }

    out_buffer[0] = PACKET_SYNC_0;
    out_buffer[1] = PACKET_SYNC_1;
    out_buffer[2] = packet_type;
    out_buffer[3] = seq_id;
    out_buffer[4] = (uint8_t)(payload_len & 0xFF);
    out_buffer[5] = (uint8_t)((payload_len >> 8) & 0xFF);

    if (payload_len > 0 && payload != NULL) {
        memcpy(&out_buffer[PACKET_HEADER_SIZE], payload, payload_len);
    }

    // CRC computed over Header + Payload
    uint16_t crc = calc_crc16(out_buffer, PACKET_HEADER_SIZE + payload_len);
    out_buffer[PACKET_HEADER_SIZE + payload_len] = (uint8_t)(crc & 0xFF);
    out_buffer[PACKET_HEADER_SIZE + payload_len + 1] = (uint8_t)((crc >> 8) & 0xFF);

    return total_len;
}

/**
 * @brief Helper: Encode Velocity Command Packet
 */
static inline uint16_t encode_velocity_packet(
    const VelocityCmdPacket &pkt,
    uint8_t seq,
    uint8_t *out_buffer,
    uint16_t max_len
) {
    return encode_packet(
        PKT_TYPE_VELOCITY_CMD,
        seq,
        &pkt,
        sizeof(VelocityCmdPacket),
        out_buffer,
        max_len
    );
}

/**
 * @brief Helper: Encode LiDAR Perception Packet
 */
static inline uint16_t encode_lidar_packet(
    const LidarPerceptionPacket &pkt,
    uint8_t seq,
    uint8_t *out_buffer,
    uint16_t max_len
) {
    return encode_packet(
        PKT_TYPE_LIDAR_PERCEPTION,
        seq,
        &pkt,
        sizeof(LidarPerceptionPacket),
        out_buffer,
        max_len
    );
}

/**
 * @brief Helper: Encode Heartbeat Packet
 */
static inline uint16_t encode_heartbeat_packet(
    const HeartbeatPacket &pkt,
    uint8_t seq,
    uint8_t *out_buffer,
    uint16_t max_len
) {
    return encode_packet(
        PKT_TYPE_HEARTBEAT,
        seq,
        &pkt,
        sizeof(HeartbeatPacket),
        out_buffer,
        max_len
    );
}

/**
 * @brief Generic function to decode a binary frame and verify CRC.
 */
static inline bool decode_packet(
    const uint8_t *buffer,
    uint16_t len,
    uint8_t *out_type,
    uint8_t *out_seq,
    uint8_t *out_payload,
    uint16_t *out_payload_len
) {
    if (len < PACKET_MIN_SIZE) {
        return false;
    }
    if (buffer[0] != PACKET_SYNC_0 || buffer[1] != PACKET_SYNC_1) {
        return false;
    }

    uint8_t p_type = buffer[2];
    uint8_t p_seq = buffer[3];
    uint16_t p_len = (uint16_t)buffer[4] | (((uint16_t)buffer[5]) << 8);

    uint16_t expected_total_len = PACKET_HEADER_SIZE + p_len + PACKET_CRC_SIZE;
    if (len < expected_total_len) {
        return false;
    }

    // Verify CRC16
    uint16_t computed_crc = calc_crc16(buffer, PACKET_HEADER_SIZE + p_len);
    uint16_t rx_crc = (uint16_t)buffer[PACKET_HEADER_SIZE + p_len] |
                      (((uint16_t)buffer[PACKET_HEADER_SIZE + p_len + 1]) << 8);

    if (computed_crc != rx_crc) {
        return false;
    }

    if (out_type) *out_type = p_type;
    if (out_seq) *out_seq = p_seq;
    if (out_payload_len) *out_payload_len = p_len;
    if (out_payload && p_len > 0) {
        memcpy(out_payload, &buffer[PACKET_HEADER_SIZE], p_len);
    }

    return true;
}

/**
 * @brief Decode VelocityCmdPacket with CRC check.
 */
static inline bool decode_velocity_packet(
    const uint8_t *buffer,
    uint16_t len,
    VelocityCmdPacket *out_cmd,
    uint8_t *out_seq
) {
    uint8_t p_type = 0;
    uint16_t p_len = 0;
    uint8_t payload[sizeof(VelocityCmdPacket)];

    if (!decode_packet(buffer, len, &p_type, out_seq, payload, &p_len)) {
        return false;
    }
    if (p_type != PKT_TYPE_VELOCITY_CMD || p_len != sizeof(VelocityCmdPacket)) {
        return false;
    }
    if (out_cmd) {
        memcpy(out_cmd, payload, sizeof(VelocityCmdPacket));
    }
    return true;
}

/**
 * @brief Decode LidarPerceptionPacket with CRC check.
 */
static inline bool decode_lidar_packet(
    const uint8_t *buffer,
    uint16_t len,
    LidarPerceptionPacket *out_lidar,
    uint8_t *out_seq
) {
    uint8_t p_type = 0;
    uint16_t p_len = 0;
    uint8_t payload[sizeof(LidarPerceptionPacket)];

    if (!decode_packet(buffer, len, &p_type, out_seq, payload, &p_len)) {
        return false;
    }
    if (p_type != PKT_TYPE_LIDAR_PERCEPTION || p_len != sizeof(LidarPerceptionPacket)) {
        return false;
    }
    if (out_lidar) {
        memcpy(out_lidar, payload, sizeof(LidarPerceptionPacket));
    }
    return true;
}

/**
 * @brief Decode HeartbeatPacket with CRC check.
 */
static inline bool decode_heartbeat_packet(
    const uint8_t *buffer,
    uint16_t len,
    HeartbeatPacket *out_hb,
    uint8_t *out_seq
) {
    uint8_t p_type = 0;
    uint16_t p_len = 0;
    uint8_t payload[sizeof(HeartbeatPacket)];

    if (!decode_packet(buffer, len, &p_type, out_seq, payload, &p_len)) {
        return false;
    }
    if (p_type != PKT_TYPE_HEARTBEAT || p_len != sizeof(HeartbeatPacket)) {
        return false;
    }
    if (out_hb) {
        memcpy(out_hb, payload, sizeof(HeartbeatPacket));
    }
    return true;
}

#ifdef __cplusplus
}
#endif

#endif // UART_PACKET_PROTOCOL_H
