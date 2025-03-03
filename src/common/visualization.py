
import cv2
import matplotlib.pyplot as plt
from common.io import read_image


def display_image(image_path=None, image=None, window_name="Image", size=None, 
                  inline=True, cmap='grey', colorbar=False, info=True):
    """
    Display an image using OpenCV or matplotlib in a Jupyter notebook.
    
    Args:
        image_path (str, optional): Path to the image file
        image (numpy.ndarray, optional): Image as a numpy array
        window_name (str): Title of the window/plot
        size (tuple, optional): Size to resize image to (width, height)
        inline (bool): If True, display inline with matplotlib, else use cv2 window
        cmap (str): Colormap for matplotlib (ignored for color images when inline=True)
        colorbar (bool): Show colorbar (only when inline=True)
    
    Returns:
        numpy.ndarray: The displayed image array
    """
    # Load image if path is provided, otherwise use the provided image array
    if image_path is not None:
        img = read_image(image_path=image_path, info=info)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None
    elif image is not None:
        img = image.copy()
    else:
        print("Error: Either image_path or image must be provided")
        return None
    
    # Get image dimensions
    if len(img.shape) == 3:
        height, width, channels = img.shape
        is_color = channels > 1
    else:
        height, width = img.shape
        is_color = False
    
    print(f"Image dimensions: {width}x{height}")
    
    # Resize if requested
    if size is not None:
        img = cv2.resize(img, size)
    
    if inline:
        # Display inline in the notebook using matplotlib
        plt.figure(figsize=(10, 8))
        
        if is_color:
            # Convert BGR to RGB for matplotlib
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_display)
        else:
            # For grayscale images, use the specified colormap
            plt.imshow(img, cmap=cmap)
            
        if colorbar:
            plt.colorbar(label='Intensity')
            
        plt.title(window_name)
        plt.axis('on')
        plt.show()
    else:
        # Display in an OpenCV window
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return None