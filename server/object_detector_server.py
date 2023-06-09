# Author: Jimmy Wu
# Date: February 2023

import argparse
import time
from datetime import datetime
from multiprocessing.connection import Listener
from pathlib import Path
from threading import Lock, Thread
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from constants import CONN_AUTHKEY
try:
    import clip
    from vild import VildDetector
except ModuleNotFoundError:
    import os
    if not os.environ.get('CONDA_DEFAULT_ENV') == 'tidybot':  # For the main tidybot env, this error is expected
        print('Could not import clip and/or VildDetector')

class ClipClassifier:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)
        self.text_features_cache = {}

    def get_text_features(self, categories):
        categories = tuple(categories)
        if categories not in self.text_features_cache:
            text = clip.tokenize([f'a photo of a {c}' for c in categories]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            self.text_features_cache[categories] = text_features
        return self.text_features_cache[categories]

    def forward(self, image_path, categories):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = self.get_text_features(categories)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity[0].cpu().numpy()

class ObjectDetectorVisualizer:
    ALPHA = 0.5
    COLOR = (78.0 / 255, 121.0 / 255, 167.0 / 255)  # Blue

    def __init__(self):
        self.figure_name = 'Object Detector'
        plt.ion()
        scale_factor = 0.35
        plt.figure(self.figure_name, figsize=(scale_factor * 18.29, scale_factor * 21.34), dpi=100)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    def visualize(self, image, output):
        plt.figure(self.figure_name)
        boxes, masks, categories, scores = output['boxes'], output['masks'], output['categories'], output['scores']
        image = torch.from_numpy(image).cuda()  # 2 ms
        if masks is not None and len(masks) > 0:
            masks = torch.from_numpy(masks).cuda()  # 5 ms
        plt.clf()  # 12 ms
        plt.axis('off')  # 7 ms
        for i, box in enumerate(boxes):
            # Draw box
            xmin, ymin, xmax, ymax = box
            plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=self.COLOR))

            # Draw mask
            if masks is not None:
                for c in range(3):
                    image[:, :, c] = torch.where(masks[i] == 1, ((1 - self.ALPHA) * image[:, :, c] + self.ALPHA * 255 * self.COLOR[c]).byte(), image[:, :, c])

            # Draw label
            plt.text(xmin, ymin, f'{categories[i]} ({scores[i]:0.2f})', fontsize=6, bbox={'facecolor': 'white', 'alpha': 0.5})

        plt.imshow(image.cpu().numpy())  # 12 ms
        plt.savefig(Path(output['image_path']).with_suffix('.png'))
        plt.pause(0.01)  # 40 ms

class ObjectDetectorServer:
    def __init__(self, hostname='0.0.0.0', port=6003, debug=False):
        self.listener = Listener((hostname, port), authkey=CONN_AUTHKEY)
        self.lock = Lock()

        # Set up ViLD object detector
        self.detector = VildDetector()

        # Set up CLIP classifier
        self.classifier = ClipClassifier()

        # Visualization
        self.debug = debug
        if self.debug:
            self.timestamp = None
            self.encoded_image = None
            self.output = None

    def forward(self, request):
        # Save JPEG image to file
        today = datetime.now().strftime('%Y-%m-%d')
        image_dir = Path(f'images/{today}')
        if not image_dir.exists():
            image_dir.mkdir(parents=True)
        if sum(1 for _ in image_dir.iterdir()) > 1000:
            print(f'Warning: {image_dir} contains over 1000 files')
        image_path = str(image_dir / f'image-{int(10 * time.time()) % 100000000}.jpg')
        with open(image_path, 'wb') as f:
            f.write(request['encoded_image'])

        with self.lock:  # Only one thread at a time can run detector
            if request.get('use_clip', False):
                # Classification
                scores = self.classifier.forward(image_path, request['categories'])
                indices = np.argsort(-scores)
                output = {
                    'boxes': [[10, 20 + 20*i, 10, 20 + 20*i] for i in range(len(indices))],  # Label placement in visualization
                    'masks': None,
                    'scores': scores[indices].tolist(),
                    'categories': [request['categories'][i] for i in indices],
                }
            else:
                # Object detection
                output = self.detector.forward(
                    image_path, request['categories'],
                    request.get('min_box_area', 220), request.get('max_box_area', float('inf')))
            self.timestamp = time.time()
            self.encoded_image = request['encoded_image']
            self.output = output
            output['image_path'] = image_path

        return output

    def handle_conn(self, conn):
        try:
            while True:
                request = conn.recv()
                output = self.forward(request)
                conn.send(output)
        except (ConnectionResetError, EOFError, BrokenPipeError):
            pass

    def handle_conns(self):
        address, port = self.listener.address
        print(f'Waiting for connections ({address}:{port})')
        while True:
            conn = self.listener.accept()
            print(f'Connected! ({address}:{port})')
            Thread(target=self.handle_conn, args=(conn,), daemon=True).start()

    def run(self):
        if not self.debug:
            self.handle_conns()
        else:
            Thread(target=self.handle_conns, daemon=True).start()

            # Tkinter GUI main loop must be in main thread
            visualizer = ObjectDetectorVisualizer()
            last_timestamp = self.timestamp
            while True:
                if self.timestamp != last_timestamp:
                    with self.lock:  # Make sure image/output are not updated while drawing visualization
                        image = cv.cvtColor(cv.imdecode(self.encoded_image, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
                        visualizer.visualize(image, self.output)
                time.sleep(0.001)

def main(args):
    ObjectDetectorServer(debug=args.debug).run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    main(parser.parse_args())
