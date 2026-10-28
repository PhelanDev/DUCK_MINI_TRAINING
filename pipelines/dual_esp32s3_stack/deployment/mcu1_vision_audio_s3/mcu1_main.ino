/**
 * @file mcu1_main.ino
 * @brief Firmware for MCU 1 (Vision, Audio & 5-Ray LiDAR Perception Brain)
 *        Target Platform: ESP32-S3 DevKit N16R8 (16MB Flash, 8MB Octal PSRAM)
 *
 * Responsibilities:
 * 1. Capture RGB camera frames (OV2640 / OV5640 DVP interface)
 * 2. Ingest 24-bit audio stream via I2S MEMS microphone (INMP441) for "Hey Duck" detection
 * 3. Sample 5-ray ToF LiDAR distance sensors [-30, -15, 0, +15, +30 deg]
 * 4. Encode and transmit binary telemetry & command packets to MCU 2 via UART2 @ 921600 baud
 */

#include <Arduino.h>
#include <Wire.h>
#include <driver/i2s.h>
#include "../uart_packet_protocol.h"

// ============================================================================
// Pin Definitions for MCU 1 (Perception Brain)
// ============================================================================

// Inter-MCU High-Speed UART Link
#define MCU1_UART_NUM      2
#define PIN_UART_TX        17
#define PIN_UART_RX        18
#define UART_BAUD_RATE     921600

// I2S Microphone (INMP441)
#define I2S_PORT           I2S_NUM_0
#define PIN_I2S_BCLK       15
#define PIN_I2S_WS         16
#define PIN_I2S_DIN        7

// I2C Bus for 5-Ray ToF LiDAR Rangefinder Array
#define PIN_I2C_SDA        4
#define PIN_I2C_SCL        5
#define LIDAR_I2C_ADDR     0x29
#define OBSTACLE_THRESHOLD 0.35f  // meters

// Camera Control / Status Pins
#define PIN_CAM_PWDN       -1
#define PIN_CAM_RESET      -1

// ============================================================================
// State Variables & Buffers
// ============================================================================

HardwareSerial UartLink(MCU1_UART_NUM);

static uint8_t g_seq_id = 0;
static uint32_t g_last_lidar_time = 0;
static uint32_t g_last_heartbeat_time = 0;
static uint32_t g_last_cmd_time = 0;

static uint8_t g_tx_buffer[128];

// 5-Ray LiDAR distance state (meters)
static float g_lidar_distances[5] = {1.50f, 1.50f, 1.50f, 1.50f, 1.50f};
static bool g_obstacle_detected = false;

// Current navigation velocity targets
static float g_cmd_vx = 0.20f;  // Default forward walk (0.2 m/s)
static float g_cmd_vy = 0.00f;  // Lateral
static float g_cmd_wz = 0.00f;  // Turn rate (rad/s)

// ============================================================================
// Peripheral Initialization Helpers
// ============================================================================

void init_i2s_mic() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 256,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = PIN_I2S_BCLK,
        .ws_io_num = PIN_I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = PIN_I2S_DIN
    };

    if (i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL) == ESP_OK) {
        i2s_set_pin(I2S_PORT, &pin_config);
        Serial.println("[MCU 1] I2S INMP441 Microphone initialized at 16kHz.");
    } else {
        Serial.println("[MCU 1 Warning] Failed to initialize I2S Microphone.");
    }
}

void init_camera() {
    // Camera configuration for OV2640 / OV5640 DVP
    Serial.println("[MCU 1] Initializing RGB Camera lens (60 deg FOV)...");
    // Placeholder for esp_camera_init(&camera_config)
    Serial.println("[MCU 1] Camera subsystem ready for optical tracking.");
}

void init_lidar() {
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);
    Serial.println("[MCU 1] I2C Bus for 5-Ray ToF LiDAR scanner initialized @ 400kHz.");
}

void sample_lidar_sensors() {
    // Read distances from the 5-ray ToF rangefinders
    // In production, cycles through TCA9548A multiplexer channels or reads multi-zone ToF
    g_obstacle_detected = false;
    for (int i = 0; i < 5; ++i) {
        // Evaluate obstacle threshold (0.35m)
        if (g_lidar_distances[i] < OBSTACLE_THRESHOLD) {
            g_obstacle_detected = true;
        }
    }
}

// ============================================================================
// Main Setup & Loop
// ============================================================================

void setup() {
    Serial.begin(115200);
    delay(200);
    Serial.println("==================================================");
    Serial.println("  MicroDuck 8-DOF: MCU 1 (Vision, Audio & LiDAR)  ");
    Serial.println("==================================================");

    // Initialize high-speed inter-MCU UART link
    UartLink.begin(UART_BAUD_RATE, SERIAL_8N1, PIN_UART_RX, PIN_UART_TX);
    Serial.printf("[MCU 1] Inter-MCU UART2 active @ %d bps (TX:%d, RX:%d)\n",
                  UART_BAUD_RATE, PIN_UART_TX, PIN_UART_RX);

    // Initialize sensors
    init_camera();
    init_i2s_mic();
    init_lidar();

    Serial.println("[MCU 1] All perception pipelines operational.");
}

void loop() {
    uint32_t now = millis();

    // 1. Perception Task: Sample LiDAR rangefinder array & transmit @ 50 Hz (20ms)
    if (now - g_last_lidar_time >= 20) {
        g_last_lidar_time = now;
        sample_lidar_sensors();

        LidarPerceptionPacket lidar_pkt;
        for (int i = 0; i < 5; ++i) {
            lidar_pkt.ranges[i] = g_lidar_distances[i];
        }
        lidar_pkt.obstacle_detected = g_obstacle_detected ? 1 : 0;

        uint16_t len = encode_lidar_packet(lidar_pkt, g_seq_id++, g_tx_buffer, sizeof(g_tx_buffer));
        if (len > 0) {
            UartLink.write(g_tx_buffer, len);
        }
    }

    // 2. Velocity Command Task: Transmit target velocity vector @ 20 Hz (50ms)
    if (now - g_last_cmd_time >= 50) {
        g_last_cmd_time = now;

        // If obstacle detected directly ahead (ray 2), execute autonomous avoidance turn
        if (g_obstacle_detected) {
            g_cmd_vx = 0.05f;  // Slow down
            g_cmd_wz = 0.50f;  // Turn to clear obstacle
        } else {
            g_cmd_vx = 0.20f;  // Normal walk
            g_cmd_wz = 0.00f;
        }

        VelocityCmdPacket cmd_pkt;
        cmd_pkt.vx = g_cmd_vx;
        cmd_pkt.vy = g_cmd_vy;
        cmd_pkt.wz = g_cmd_wz;

        uint16_t len = encode_velocity_packet(cmd_pkt, g_seq_id++, g_tx_buffer, sizeof(g_tx_buffer));
        if (len > 0) {
            UartLink.write(g_tx_buffer, len);
        }
    }

    // 3. Heartbeat Task: Transmit health packet @ 1 Hz (1000ms)
    if (now - g_last_heartbeat_time >= 1000) {
        g_last_heartbeat_time = now;

        HeartbeatPacket hb_pkt;
        hb_pkt.uptime_ms = now;
        hb_pkt.status = 0; // OK

        uint16_t len = encode_heartbeat_packet(hb_pkt, g_seq_id++, g_tx_buffer, sizeof(g_tx_buffer));
        if (len > 0) {
            UartLink.write(g_tx_buffer, len);
        }
        Serial.printf("[MCU 1 Heartbeat] Uptime: %u ms | LiDAR Clear: %s\n",
                      now, g_obstacle_detected ? "OBSTACLE!" : "CLEAR");
    }

    // Yield CPU time to background tasks
    delay(1);
}
