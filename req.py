import requests

url = 'http://localhost:5000/depth'
image_path = './image.png'

with open(image_path, 'rb') as img_file:
    files = {'image': img_file}
    response = requests.post(url, files=files)

if response.status_code == 200:
    depth_filename = image_path.split('/')[-1].split('.')[0] + '_depth.png'
    with open(depth_filename, 'wb') as f:
        f.write(response.content)
    print('Depth map saved as depth_map.png')
else:
    print('Error:', response.json())
