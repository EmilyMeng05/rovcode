import argparse
import asyncio
import json
import logging
import cv2
import numpy as np
import websockets
from websockets.client import ClientConnection

# Initialize ORB detector for visual odometry
orb = cv2.ORB_create()

def main():
    print('onboard client starting')
    rov, websocket_uri = setup_using_command_line_args()
    asyncio.run(client_handler(websocket_uri, rov))


# allows file to be run with arguments
# includes default websocket if none specified 
def setup_using_command_line_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-p', '--physical', action='store_true',
                            help='run using physical instead of simulated hardware')
    arg_parser.add_argument('-w', '--websocket', default='localhost:8001',
                            help='websocket\'s ip and port, e.g. localhost:8001')
    args = arg_parser.parse_args()

    if args.physical:
        from physical.physical import ROV
    else:
        from simulated.simulated import ROV

    rov = ROV()
    websocket_uri = f'ws://{args.websocket}'
    
    return rov, websocket_uri


# connect to surface station
async def client_handler(websocket_uri: str, rov: 'ROV'):
    async for websocket in websockets.connect(websocket_uri):
        try:
            await asyncio.gather(
                consumer_handler(websocket, rov),
                producer_handler(websocket, rov)
            )
        except websockets.ConnectionClosed:
            print('onboard: websockets.ConnectionClosed - retrying...')
            await asyncio.sleep(1)
            continue


# incoming command handling (instructions for ROV)
async def consumer_handler(websocket: ClientConnection, rov: 'ROV'):
    async for message in websocket:
        data = json.loads(message)
        if data['type'] == 'set_pin_pwms':
            for pin in data['pins']:
                rov.set_pin_pwm(pin['number'], pin['value'])
            await rov.flush_pin_pwms()
        else:
            print(f'Invalid type {data.type} for message: {message}')


# outgoing data handling (sensor readings and odometry)
async def producer_handler(websocket: ClientConnection, rov: 'ROV'):
    while True:
        await asyncio.gather(
            send_onboard_digest(websocket, rov),
            asyncio.sleep(0.1)  # limit to 10 summaries per second
        )


# Send sensor data and odometry to the surface station
async def send_onboard_digest(websocket: ClientConnection, rov: 'ROV'):
    # Poll sensors (e.g., IMU)
    gyro, accel = await rov.poll_sensors()

    # Capture frame from the ROV's camera
    frame = rov.get_camera_frame()

    # Compute odometry (if a previous frame exists)
    odometry_data = None
    if rov.prev_frame is not None:
        # Detect keypoints and descriptors
        prev_kp, prev_desc = orb.detectAndCompute(rov.prev_frame, None)
        curr_kp, curr_desc = orb.detectAndCompute(frame, None)

        # Match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(prev_desc, curr_desc)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([curr_kp[m.trainIdx].pt for m in matches])

        # Compute essential matrix and recover pose
        E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2)

        # Store odometry data
        odometry_data = {
            'rotation': R.tolist(),  # Convert numpy array to list
            'translation': t.tolist()
        }

    # Update previous frame
    rov.prev_frame = frame

    # Send sensor and odometry data
    await websocket.send(json.dumps({
        'type': 'sensor_summary',
        'accelerometer': accel,
        'gyroscope': gyro,
        'odometry': odometry_data
    }))


if __name__ == "__main__":
    main()