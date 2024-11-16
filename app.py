from flask import Flask, request, jsonify, send_file
from realesrgan import RealESRGAN
import os

app = Flask(__name__)

# Initialize Real-ESRGAN model
model = RealESRGAN('RealESRGAN_x4plus')

@app.route('/enhance', methods=['POST'])
def enhance_image():
    try:
        # Save the uploaded image
        file = request.files['image']
        input_path = 'input.jpg'
        file.save(input_path)

        # Enhance the image
        output_path = 'output.jpg'
        model.enhance(input_path, output_path)

        # Send the enhanced image back
        return send_file(output_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
 
        
