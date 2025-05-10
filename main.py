
from utils.streaming import VStream
from utils.tracking import track
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
os.environ['YOLO_VERBOSE'] = 'False'

# 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device('cpu')

model = torch.load('models/model.pt', map_location=device)

class_names = {
    0: 'calling',
    1: 'clapping',
    2: 'cycling',
    3: 'dancing',
    4: 'drinking',
    5: 'eating',
    6: 'fighting',
    7: 'hugging',
    8: 'laughing',
    9: 'listening_to_music',
    10: 'running',
    11: 'sitting',
    12: 'sleeping',
    13: 'texting',
    14: 'using_laptop'
}

vstream = VStream(width=1024, height=512)
vstream.start()

transform = transforms.Compose([
    # transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

win_title = "Frame"

while True:
    coords, state = [], []
    the_frame = vstream.read()

    boxes = track(the_frame, persist=False)
    a = boxes.cpu().numpy().copy()
    nrows, _ = a.shape
    r = (1, 1)
    for i in range(nrows):
        x1, y1, x2, y2 = (a[i][:]).astype(int)
        coord = (int((x1+x2)/2), int(y1*r[1]))
        coords.append(coord)

        orgx, orgy = int((x1+x2)/2), int((y1+y2)/2)
        # crop_img = the_frame[orgy-300:orgy+300, orgx-300:orgx+300]
        crop_img = the_frame[y1:y2, x1:x2]
        # crop_img = cv2.resize(crop_img, (256, 256))

        image = Image.fromarray(crop_img)
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        with torch.no_grad():
            outputs = model(img)
            _, st = torch.max(outputs, 1)
        state.append(st[0].item())
        print("State:", class_names[state[0]])

        for st, coord in zip(state, coords):
            the_frame = cv2.putText(
                the_frame, class_names[st], coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 1)

        cv2.imshow(win_title, the_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or cv2.getWindowProperty(win_title,
                                                cv2.WND_PROP_VISIBLE) < 1:
        break

# %%
# Release the video capture
vstream.stream.release()
cv2.destroyAllWindows()
