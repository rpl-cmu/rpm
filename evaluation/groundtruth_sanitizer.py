import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tyro
import pickle
import numpy as np
from PIL import Image

def pkl_to_png(directory_path, output_directory=None):
    """
    Convert all .pkl files in a directory to PNG images.
    
    Args:
        directory_path (str): Path to directory containing .pkl files
        output_directory (str): Optional output directory. If None, saves in same directory as .pkl files
    """
    
    # Use input directory as output if not specified
    if output_directory is None:
        output_directory = directory_path
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Find all .pkl files in the directory
    pkl_files = [f for f in os.listdir(directory_path) if f.endswith('.pkl')]
    
    if not pkl_files:
        print(f"No .pkl files found in {directory_path}")
        return
    
    print(f"Found {len(pkl_files)} .pkl files to process...")
    
    for pkl_file in pkl_files:
        try:
            # Load the pickle file
            pkl_path = os.path.join(directory_path, pkl_file)
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # Generate output filename (replace .pkl with .png)
            png_filename = pkl_file.replace('.pkl', '.png')
            png_path = os.path.join(output_directory, png_filename)
            
            # Handle different data types
            if isinstance(data, dict) and 'data' in data:
                # If it's a dict with 'data' key, extract that
                image_data = data['data']
            else:
                # Otherwise use the data directly
                image_data = data
            
            # Convert to numpy array if not already
            if not isinstance(image_data, np.ndarray):
                image_data = np.array(image_data)
            
            # Handle different array shapes and types
            if len(image_data.shape) == 1:
                # 1D array - reshape to square if possible
                size = int(np.sqrt(len(image_data)))
                if size * size == len(image_data):
                    image_data = image_data.reshape(size, size)
                else:
                    # If not perfect square, create a horizontal strip
                    image_data = image_data.reshape(1, -1)
            
            # Normalize data to 0-255 range if needed
            if image_data.dtype != np.uint8:
                if image_data.max() <= 1.0 and image_data.min() >= 0.0:
                    # Data is in 0-1 range
                    image_data = (image_data * 255).astype(np.uint8)
                else:
                    # Normalize to 0-255 range
                    image_data = ((image_data - image_data.min()) / 
                                 (image_data.max() - image_data.min()) * 255).astype(np.uint8)
            
            # Save as PNG
            if len(image_data.shape) == 2:
                # Grayscale image
                img = Image.fromarray(image_data, mode='L')
            elif len(image_data.shape) == 3 and image_data.shape[2] == 3:
                # RGB image
                img = Image.fromarray(image_data, mode='RGB')
            elif len(image_data.shape) == 3 and image_data.shape[2] == 4:
                # RGBA image
                img = Image.fromarray(image_data, mode='RGBA')
            else:
                # For other shapes, use matplotlib to create a visualization
                plt.figure(figsize=(10, 8))
                plt.imshow(image_data, cmap='viridis')
                plt.colorbar()
                plt.title(f"Data from {pkl_file}")
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Converted {pkl_file} → {png_filename} (using matplotlib)")
                continue
            
            img.save(png_path)
            print(f"Converted {pkl_file} → {png_filename}")
            
        except Exception as e:
            print(f"Error processing {pkl_file}: {str(e)}")
            continue
    
    print(f"Conversion complete! PNG files saved in {output_directory}")


def replace_pkl_data_with_png(png_path, pkl_path, output_pkl_path=None):
    """
    Load a PNG image and a PKL file, replace the 'data' field in the PKL 
    with the PNG data, and save the modified PKL file.
    
    Args:
        png_path (str): Path to the PNG file
        pkl_path (str): Path to the input PKL file
        output_pkl_path (str, optional): Path to save the modified PKL file.
                                       If None, overwrites the original PKL file.
    
    Returns:
        dict: The modified data structure that was saved
    """
    
    # Load the PNG image
    try:
        image = Image.open(png_path)
        # Convert to numpy array (you can modify this based on your needs)
        png_data = np.array(image)
        print(f"Loaded PNG: {png_path} with shape {png_data.shape}")
    except Exception as e:
        raise ValueError(f"Error loading PNG file {png_path}: {e}")
    
    # Load the PKL file
    try:
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        print(f"Loaded PKL file: {pkl_path}")
    except Exception as e:
        raise ValueError(f"Error loading PKL file {pkl_path}: {e}")
    
    # Check if pkl_data is a dictionary and has a 'data' key
    if not isinstance(pkl_data, dict):
        raise ValueError("PKL file must contain a dictionary")
    
    # Replace the 'data' field with PNG data
    pkl_data['data'] = png_data[:, :, 0]
    print(f"Replaced 'data' field with PNG data")
    
    # Determine output path
    if output_pkl_path is None:
        output_pkl_path = pkl_path
    
    # Save the modified PKL file
    try:
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(pkl_data, f)
        print(f"Saved modified PKL file to: {output_pkl_path}")
    except Exception as e:
        raise ValueError(f"Error saving PKL file {output_pkl_path}: {e}")
    
    return pkl_data

if __name__ == "__main__":
    '''
    This is a two stage work process:
    1. Convert all .pkl files in a directory to PNG images.
    2. Use image editors to modify the PNG images in some other software.
    3. Replace the 'data' field in the PKL files with the modified PNG data.

    If I am a better person, I would have set this script up with a flag on whether to convert or replace.
    But I am not, so I will just run the replace function directly.
    You can also use the pkl_to_png function to convert all .pkl files in a directory to PNG images.
    You can also use the replace_pkl_data_with_png function to replace the 'data' field in a PKL file with the PNG data.

    Note: Copilot wrote the above paragraph.
    '''
    file_name = "square1"

    png_path = f"data/collected_data/map_lidar_sanitized/{file_name}.png"
    pkl_path = f"data/collected_data/map_raw_lidar/{file_name}.pkl"
    output_pkl = f"data/collected_data/map_lidar_sanitized/{file_name}.pkl"

    replace_pkl_data_with_png(png_path, pkl_path, output_pkl)

    # Other usages:
    # pkl_to_png(input_directory, output_dir)
