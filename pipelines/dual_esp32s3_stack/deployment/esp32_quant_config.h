#ifndef ESP32_QUANT_CONFIG_H
#define ESP32_QUANT_CONFIG_H

/**
 * MicroDuck 8-DOF (Dual ESP32-S3 Stack) - INT8 Quantization & Dimension Config
 * Automatically generated for Xtensa LX7 PIE Vector Accelerator.
 */

#define OBS_DIM 42
#define ACTION_DIM 8
#define LIDAR_RAYS 5

#define OBS_SCALE 127.0f
#define ACT_SCALE 127.0f
#define OBS_INV_SCALE (1.0f / 127.0f)
#define ACT_INV_SCALE (1.0f / 127.0f)

// Physical maximum normalization bounds for 42 observation dimensions
static const float OBS_MAX_BOUNDS[OBS_DIM] = {
    6.00f, 6.00f, 6.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.57f, 0.78f, 0.44f, 0.87f, 1.48f, 0.44f, 0.87f, 1.48f, 6.00f, 6.00f, 6.00f, 6.00f, 6.00f, 6.00f, 6.00f, 6.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 2.00f, 2.00f, 2.00f, 2.00f, 2.00f
};

// Effective INT8 scale per observation channel: (scale_obs / OBS_MAX_BOUNDS[i])
static const float OBS_CHANNEL_SCALES[OBS_DIM] = {
    21.1667f, 21.1667f, 21.1667f, 127.0000f, 127.0000f, 127.0000f, 127.0000f, 127.0000f, 127.0000f, 80.8917f, 162.8205f, 288.6364f, 145.9770f, 85.8108f, 288.6364f, 145.9770f, 85.8108f, 21.1667f, 21.1667f, 21.1667f, 21.1667f, 21.1667f, 21.1667f, 21.1667f, 21.1667f, 127.0000f, 127.0000f, 127.0000f, 127.0000f, 127.0000f, 127.0000f, 127.0000f, 127.0000f, 127.0000f, 127.0000f, 127.0000f, 127.0000f, 63.5000f, 63.5000f, 63.5000f, 63.5000f, 63.5000f
};

// Servo Joint Angle Boundaries for MG996R (Radians)
#define HEAD_YAW_MIN -1.57f
#define HEAD_YAW_MAX 1.57f
#define HEAD_PITCH_MIN -0.52f
#define HEAD_PITCH_MAX 0.78f
#define LEFT_HIP_ROLL_MIN -0.44f
#define LEFT_HIP_ROLL_MAX 0.44f
#define LEFT_HIP_PITCH_MIN -0.87f
#define LEFT_HIP_PITCH_MAX 0.87f
#define LEFT_KNEE_PITCH_MIN 0.0f
#define LEFT_KNEE_PITCH_MAX 1.48f
#define RIGHT_HIP_ROLL_MIN -0.44f
#define RIGHT_HIP_ROLL_MAX 0.44f
#define RIGHT_HIP_PITCH_MIN -0.87f
#define RIGHT_HIP_PITCH_MAX 0.87f
#define RIGHT_KNEE_PITCH_MIN 0.0f
#define RIGHT_KNEE_PITCH_MAX 1.48f

#endif // ESP32_QUANT_CONFIG_H
