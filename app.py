import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageOps
import io
import tensorflow as tf
import time  # For measuring time
from tabulate import tabulate  # For displaying timing information as a table
import base64

app = Flask(__name__)

MODEL_PATH = 'Midas-V2.tflite'

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_HEIGHT = 256
INPUT_WIDTH = 256
INPUT_MEAN = 127.5
INPUT_STD = 127.5


from PIL import Image, ImageOps

# Preprocess image to match model input size and normalize
def preprocess_image(image: Image.Image) -> np.ndarray:
    original_size = image.size  # Store the original image size

    # Compress the image if its size is greater than 1 MB
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    size_in_mb = len(img_byte_arr.getvalue()) / (1024 * 1024)

    # If the image size exceeds 1 MB, we compress it
    if size_in_mb > 1.0:
        quality = int(min(85, 85 * (2.0 / size_in_mb)))  # Adjust quality based on size
        compressed_io = io.BytesIO()
        image.save(compressed_io, format='JPEG', quality=quality, optimize=True)
        image = Image.open(compressed_io)

    # Convert the image to RGB and resize to the input size
    image = image.convert('RGB')
    image = image.resize((INPUT_WIDTH, INPUT_HEIGHT))

    # Convert the image to a NumPy array and normalize
    image_np = np.array(image).astype(np.float32)
    image_np = (image_np - INPUT_MEAN) / INPUT_STD  # Normalize
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension

    return image_np, original_size

# Postprocess depth map to match original size
def postprocess_depth(depth: np.ndarray, original_size: tuple) -> Image.Image:
    # Squeeze the depth array to remove any singleton dimensions
    depth = np.squeeze(depth)
    
    # Normalize the depth values to the range [0, 1]
    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    
    # Convert to a 0-255 range for visualization
    depth_image = (depth_normalized * 255).astype(np.uint8)

    # Convert the depth map into a PIL Image
    depth_pil = Image.fromarray(depth_image)

    # Resize the depth map back to the original size
    depth_pil = depth_pil.resize(original_size, Image.Resampling.LANCZOS)

    return depth_pil




def calculate_patch_values(depth_image: Image.Image) -> dict:
    depth_array = np.array(depth_image)
    h, w = depth_array.shape
    patch_values = {}
    patch_size_h = h // 4
    patch_size_w = w // 4

    for i in range(4):
        for j in range(4):
            patch = depth_array[i * patch_size_h:(i + 1) * patch_size_h,
                                j * patch_size_w:(j + 1) * patch_size_w]
            patch_values[4 * i + j + 1] = int(np.mean(patch))

    return patch_values

def print_patch_values_matrix(patch_values: dict):
    matrix = np.zeros((4, 4), dtype=int)
    for patch, value in patch_values.items():
        row = (patch - 1) // 4
        col = (patch - 1) % 4
        matrix[row, col] = value
    print("\nPatch Values Matrix:")
    print(matrix)

import numpy as np


def generate_response(patch_values: dict) -> str:
    # Print patch values matrix in console
    print_patch_values_matrix(patch_values)

    # Obstacle Detection Logic
    obstacle_patches = []
    obstacle_details = []

    # Left Side Detection (Divide into 3 parts: upper left, middle left, lower left)
    left_avg = np.mean([patch_values[1], patch_values[5], patch_values[9], patch_values[13]])
    if left_avg > 75:  # Add "left" if the overall average is high
        if "left" not in obstacle_patches:
            obstacle_patches.append("left")
            obstacle_details.append("left side")
    else:
        # Check individual subregions only if overall "left" is not obstructed
        upper_left_avg = np.mean([patch_values[1], patch_values[5]])
        if upper_left_avg > 100 and "upper left" not in obstacle_patches:
            obstacle_patches.append("upper left")
            obstacle_details.append("upper left side")

        middle_left_avg = np.mean([patch_values[5], patch_values[9]])
        if middle_left_avg > 100 and "middle left" not in obstacle_patches:
            obstacle_patches.append("middle left")
            obstacle_details.append("middle left side")

        lower_left_avg = np.mean([patch_values[9], patch_values[13]])
        if lower_left_avg > 100 and "lower left" not in obstacle_patches:
            obstacle_patches.append("lower left")
            obstacle_details.append("lower left side")

    # Right Side Detection (Divide into 3 parts: upper right, middle right, lower right)
    right_avg = np.mean([patch_values[4], patch_values[8], patch_values[14], patch_values[16]])
    if right_avg > 75:  # Add "right" if the overall average is high
        if "right" not in obstacle_patches:
            obstacle_patches.append("right")
            obstacle_details.append("right side")
    else:
        # Check individual subregions only if overall "right" is not obstructed
        upper_right_avg = np.mean([patch_values[4], patch_values[8]])
        if upper_right_avg > 100 and "upper right" not in obstacle_patches:
            obstacle_patches.append("upper right")
            obstacle_details.append("upper right side")

        middle_right_avg = np.mean([patch_values[8], patch_values[14]])
        if middle_right_avg > 100 and "middle right" not in obstacle_patches:
            obstacle_patches.append("middle right")
            obstacle_details.append("middle right side")

        lower_right_avg = np.mean([patch_values[14], patch_values[16]])
        if lower_right_avg > 100 and "lower right" not in obstacle_patches:
            obstacle_patches.append("lower right")
            obstacle_details.append("lower right side")

    # Center Detection (Divide into 3 parts: upper center, middle center, lower center)
    center_avg = np.mean([patch_values[6], patch_values[7], patch_values[10], patch_values[11]])
    if center_avg > 120:  # Add "center" if the overall average is high
        if "center" not in obstacle_patches:
            obstacle_patches.append("center")
            obstacle_details.append("center")
    else:
        # Check individual subregions only if overall "center" is not obstructed
        upper_center_avg = np.mean([patch_values[6], patch_values[7]])
        if upper_center_avg > 120 and "upper center" not in obstacle_patches:
            obstacle_patches.append("upper center")
            obstacle_details.append("upper center")

        middle_center_avg = np.mean([patch_values[7], patch_values[10]])
        if middle_center_avg > 120 and "middle center" not in obstacle_patches:
            obstacle_patches.append("middle center")
            obstacle_details.append("middle center")

        lower_center_avg = np.mean([patch_values[10], patch_values[11]])
        if lower_center_avg > 120 and "lower center" not in obstacle_patches:
            obstacle_patches.append("lower center")
            obstacle_details.append("lower center")

    # Overall Left or Right Obstacle Detection (for redundancy check)
    overall_left_avg = np.mean([patch_values[1], patch_values[2], patch_values[5], patch_values[6], patch_values[9], patch_values[10], patch_values[13], patch_values[14]])
    if overall_left_avg >= 120 and "left" not in obstacle_patches:
        obstacle_patches.append("left")
        obstacle_details.append("left side")
    
    overall_right_avg = np.mean([patch_values[3], patch_values[4], patch_values[7], patch_values[8], patch_values[11], patch_values[12], patch_values[15], patch_values[16]])
    if overall_right_avg >= 120 and "right" not in obstacle_patches:
        obstacle_patches.append("right")
        obstacle_details.append("right side")

    # Determining the direction of obstacles with exact locations
    if obstacle_patches:
        obstacle_location_str = "There appears to be an object in the " + ", ".join(obstacle_details) + "."
    else:
        obstacle_location_str = "No obstacles detected."

    # Clear path detection
    clear_direction = []
    if "left" not in obstacle_patches:
        clear_direction.append("left")
    if "right" not in obstacle_patches:
        clear_direction.append("right")
    if "center" not in obstacle_patches:
        clear_direction.append("center")

    clear_statement = f"Area directly to the {' and '.join(clear_direction)} seems to be clear." if clear_direction else "No clear direction found."

    return f"{obstacle_location_str} {clear_statement}"

@app.route('/depth', methods=['POST'])
def generate_depth():
    overall_start = time.time()  # Start overall timing

    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    try:
        timings = []

        # Step 1: Load the image and correct orientation
        step_start = time.time()
        image = Image.open(file.stream)  # Open the image
        image = ImageOps.exif_transpose(image)  # Correct orientation if the image has EXIF data
        timings.append(["Image Loading", time.time() - step_start])

        # Step 2: Preprocess the image
        step_start = time.time()
        input_data, original_size = preprocess_image(image)
        timings.append(["Preprocessing", time.time() - step_start])

        # Step 3: Model inference
        step_start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        timings.append(["Model Inference", time.time() - step_start])

        # Step 4: Extract depth map
        step_start = time.time()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        depth = output_data[0]
        timings.append(["Depth Map Extraction", time.time() - step_start])

        # Step 5: Postprocess depth map
        step_start = time.time()
        depth_image = postprocess_depth(depth, original_size)
        timings.append(["Postprocessing", time.time() - step_start])

        # Step 6: Calculate patch values
        step_start = time.time()
        patch_values = calculate_patch_values(depth_image)
        timings.append(["Patch Value Calculation", time.time() - step_start])

        # Step 7: Generate response
        step_start = time.time()
        response = generate_response(patch_values)
        timings.append(["Response Generation", time.time() - step_start])

        overall_end = time.time()
        timings.append(["Overall Request Handling", overall_end - overall_start])

        # Print timing and response to the console
        print("\nTiming Information:")
        print(tabulate(timings, headers=["Step", "Time (seconds)"], tablefmt="grid"))
        print("\nGenerated Response:")
        print(response)

        # Send response
        img_io = io.BytesIO()
        depth_image.save(img_io, 'PNG', optimize=True, compress_level=9)
        img_io.seek(0)

        if request.headers.get('Accept') == 'application/json':
            img_base64 = base64.b64encode(img_io.getvalue()).decode()
            return jsonify({
                'depth_map': img_base64,
                'analysis': response,
                'patch_values': patch_values
            })
        else:
            return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Depth Map Generator</title>
    <h1>Final Upload an image to get its depth map</h1>
    <form method=post enctype=multipart/form-data action="/depth">
      <input type=file name=image>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)