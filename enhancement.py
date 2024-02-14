import cv2
import numpy as np
import argparse
import yaml
import os 
from diffusers import LDMSuperResolutionPipeline
import torch 
from PIL import Image
from face_tracker import FaceTracker
from insightface.app import FaceAnalysis
from transparent_background import Remover
from super_image import EdsrModel, ImageLoader
from PIL import Image
import requests

def read_single_image(image_path):
    # Read a single image
    image = cv2.imread(image_path)

    # Get the name of the image
    image_name = os.path.basename(image_path)

    return image, image_name

def read_bulk_images(directory_path):
    images = []
    image_names = []

    # Read multiple images from a specified directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
            image_path = os.path.join(directory_path, filename)
            
            # Read each image
            image = cv2.imread(image_path)
            # image = image[:, int(image.shape[0]/2):, int(image.shape[1]/2) :]

            # Get the name of the image
            image_name = filename

            # Append the image and its name to the lists
            images.append(image)
            image_names.append(image_name)

    return images, image_names   

def read_image_path(path, img_array_name = "default.png"):

    # print (path)
    if not isinstance(path, np.ndarray) :
         _, extension = os.path.splitext(path.lower())
    if  isinstance(path, np.ndarray):
        
        return path, img_array_name, "single" 
    
    elif os.path.isdir(path):
        images, images_name = read_bulk_images(path)
        
        return  images, images_name, "bulk"
       
    elif extension in ['.png', '.jpg', '.jpeg']:
        
        images, images_name = read_single_image(path)
        
        return images, images_name, "single"
    else:
        print("Invalid path. Please provide a valid image file path or directory path.")
        return None, None
  
def get_single_image_array(image_array):
    pass

class Enhancement:
    def __init__(self, inputs_path, config_file_path, img_array_name="default.png"):
        # print("byebyebyebye")
        self.image, self.image_name, self.mode = read_image_path(inputs_path, img_array_name)
        with open(config_file_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

    #add neccessary methods to config file with order you want
    #TODO:DOCUMENTATION POINT>> Order of methods list is important

    def apply_enhancement_methods(self, ):
        
        if self.mode == "single":
            for method in self.config["Enhancement"]["using_methods"]:
                for i in range(len(self.config["Enhancement"]["enhancement_methods"])):
                        if self.config["Enhancement"]["enhancement_methods"][i]['name'] == method:
                            method_info = self.config["Enhancement"]["enhancement_methods"][i]
                method_name = method_info['name']
                method_params = method_info.get('params', {})
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    self.image = method(self.image,self.image_name, **method_params)
                else:
                    print(f"Warning: Method '{method_name}' does not exist in the Preprocess class.")
            # print("yessssssssssss")
            # im_name,_ = os.path.splitext(self.image_name)
            # ratio = .25
            # new_width = int(self.image.shape[0] * ratio)
            # new_height = int(self.image.shape[0] * ratio)   
            # resized_image = self.image.resize((new_width, new_height), Image.BICUBIC)
     
            save_path = os.path.join("./data/outputs", f"{self.image}_fullyenhanced.png")                 
            self.save_image(self.image, save_path)
            return self.image

        elif self.mode == "bulk":
            for img_array, img_name in zip(self.image, self.image_name):
               for method in self.config["Enhancement"]["using_methods"]:
                    for i in range(len(self.config["Enhancement"]["enhancement_methods"])):
                        if self.config["Enhancement"]["enhancement_methods"][i]['name'] == method:
                            method_info = self.config["Enhancement"]["enhancement_methods"][i]
                    
                    method_name = method_info['name']
                    method_params = method_info.get('params', {})
                    if hasattr(self, method_name):
                        method = getattr(self, method_name)
                        img_array = method(img_array,img_name, **method_params)
                        
                    else:
                        print(f"Warning: Method '{method_name}' does not exist in the Preprocess class.")

               im_name,_ = os.path.splitext(img_name)        
               save_path = os.path.join("./data/enhanced", f"{im_name}_fullyenhanced.png")                 
               self.save_image(img_array, save_path)

    def _brightness_adjustmnet(self, image_array, image_name, **params):
        #Image Histogram
        hist = cv2.calcHist([image_array], [0], None, [256], [0, 256]) 
        hist_flat = hist.flatten()
        brightness = np.sum(hist_flat * np.arange(256)) / np.sum(hist_flat)
        
        # Adjust brightness if needed
        if brightness < params["target_brightness"]:
            adjusted_image = np.clip(image_array * params["adjustment_factor"], 0, 255).astype(np.uint8)
        elif brightness > params["target_brightness"]:
            adjusted_image = np.clip(image_array / params["adjustment_factor"], 0, 255).astype(np.uint8)
        else:
            adjusted_image = image_array.copy()

        
        if params["image_save"]:     
            im_name,_ = os.path.splitext(image_name)        
            save_path = os.path.join("./data/outputs",f"{im_name}_brighted.png")                 
            self.save_image(adjusted_image, save_path)

        return adjusted_image

    def _histogram_equalization(self, image_array, image_name, **params):
        # Apply histogram equalization
        yuv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2YUV)
        yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])
        enhanced_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
        
        if params["image_save"]:
            im_name,_ = os.path.splitext(image_name)        
            save_path = os.path.join("./data/outputs", f"{im_name}_histq.png")                 
            self.save_image(enhanced_image, save_path)

        return enhanced_image

    def _contrast_stretching(self, image_array, image_name, **params):
        # Clip pixel values to the specified threshold
        clipped_image = np.clip(image_array, 0, params["threshold"])

        # Perform contrast stretching on the clipped image
        stretched_image = (255 / params["threshold"]) * clipped_image

        # Combine the stretched and clipped parts
        result_image = np.where(image_array <= params["threshold"], stretched_image, image_array)

        if params["image_save"]:
            im_name,_ = os.path.splitext(image_name)        
            save_path = os.path.join("./data/outputs", f"{im_name}_stretched.png")                 
            self.save_image(result_image.astype(np.uint8), save_path)

        return result_image.astype(np.uint8)
    
    def _contrast_enhancement(self, image_array, image_name, **params):
        lookUpTable = np.empty((1,256), np.uint8)
        gamma = 1.2
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i/255.0, gamma) * 255.0, 0, 255)

        out = cv2.LUT(image_array, lookUpTable)

        if params["image_save"]:
            im_name,_ = os.path.splitext(image_name)        
            save_path = os.path.join("./data/outputs", f"{im_name}_contrast_enhancement.png")                 
            self.save_image(out, save_path)
        
        return out
    
    def _increase_contrast_dark_pixels(self, image_array, image_name, **params):
        # Convert the image to grayscale if it's a color image
        dark_factor=1.0
        light_factor=1.0
        if len(image_array.shape) == 3:
            image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Apply contrast stretching
        contrast_stretched = cv2.normalize(image, None, alpha=dark_factor, beta=light_factor, norm_type=cv2.NORM_MINMAX)
        if params["image_save"]:
            im_name,_ = os.path.splitext(image_name)        
            save_path = os.path.join("./data/outputs", f"{im_name}_contrast_enhancement.png")                 
            self.save_image(contrast_stretched, save_path)
         

        return contrast_stretched
    
    def _noise_reduction(self, image_array,image_name, **params):

        # Choose the noise reduction method
        if params['noise_reduction_method'] == 'original':
            preprocessed_image = image_array
        elif params['noise_reduction_method'] == 'gaussian_blur':
            preprocessed_image = cv2.GaussianBlur(image_array, (0, 0), params['gaussian_blur_sigma'])
        elif params['noise_reduction_method'] == 'median_blur':
            preprocessed_image = cv2.medianBlur(image_array, params['median_blur_kernel_size'])
        elif params['noise_reduction_method'] == 'bilateral_filter':
            preprocessed_image = cv2.bilateralFilter(image_array, params['bilateral_d'], params['bilateral_sigma_color'], params['bilateral_sigma_space'])
        elif params['noise_reduction_method'] == 'morphological_operations':
            kernel = np.ones((params['morphological_kernel_size'], params['morphological_kernel_size']), np.uint8)
            preprocessed_image = cv2.dilate(cv2.erode(image_array, kernel, iterations=1), kernel, iterations=1)
        elif params['noise_reduction_method'] == 'adaptive_thresholding':
            preprocessed_image = cv2.adaptiveThreshold(
                image_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                params['adaptive_threshold_block_size'], params['adaptive_threshold_c']
            )
        else:
            raise ValueError("Invalid noise reduction method. Choose one of: original, gaussian_blur, median_blur, "
                            "bilateral_filter, morphological_operations, adaptive_thresholding")
        
        if params["image_save"]:
            im_name,_ = os.path.splitext(image_name)        
            save_path = os.path.join("./data/outputs", f"{im_name}_denoised_{params['noise_reduction_method']}.png")                 
            self.save_image(preprocessed_image, save_path)

        return preprocessed_image
    
    def _unsharp_masking(self, image_array, image_name, **params):
        # Assuming 'image' is your input image
        blurred = cv2.GaussianBlur(image_array, params["kernel_size"], params["std"]) 
        #1.5 and -0.5 are weights
        sharpened = cv2.addWeighted(image_array,params["orginal_image_weight"], blurred, params["blurred_weight"], params["scalar"]) 
        
        if params["image_save"]:
            im_name,_ = os.path.splitext(image_name)        
            save_path = os.path.join("./data/outputs", f"{im_name}_unsharp_mask_sharpened.png")                 
            self.save_image(sharpened, save_path)

        return sharpened
    
    def _high_pass_filter(self, image_array, image_name, **params):
        laplacian = cv2.Laplacian(image_array, cv2.CV_64F)
        sharpened = cv2.addWeighted(image_array.astype(np.float64),params["orginal_image_weight"], laplacian, params["laplacian_weight"], params["scalar"]) 
        sharpened_8 = (sharpened * 255).astype(np.uint8)
        if params["image_save"]:
            im_name,_ = os.path.splitext(image_name)        
            save_path = os.path.join("./data/outputs", f"{im_name}_highpass_sharpened.png")                 
            self.save_image(sharpened_8, save_path)
        return sharpened_8

    def _enlarge_image(self, image_array, image_name, **params):
        
        # Get the new dimensions after scaling
        new_width = int(image_array.shape[1] * params['scale_factor'])
        new_height = int(image_array.shape[0] * params['scale_factor'])

        # Resize the image using bicubic interpolation
        enlarged_image = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        if params["image_save"]:
            im_name,_ = os.path.splitext(image_name)        
            save_path = os.path.join("./data/outputs", f"{im_name}_enlarged_image.png")                 
            self.save_image(enlarged_image, save_path)

        return enlarged_image 
          
    def _gray_scale(self, image_array, image_name, **params):
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        return gray_image
    
    def _binarization(self, image_array, image_name, **params):
        threshold_value = 127

        # Apply binary thresholding
        image_array_uint8 = image_array.astype(np.uint8)
        _, binary_image = cv2.threshold(image_array_uint8, threshold_value, 255, cv2.THRESH_BINARY)
        return binary_image

    def _super_resolution(self, image_array, image_name, **params):
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        model_id = "CompVis/ldm-super-resolution-4x-openimages"
        # load model and scheduler
        pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
        pipeline = pipeline.to(device)
        # run pipeline in inference (sample random noise and denoise)
        
        
        pil_image = Image.fromarray(image_array)
        # print("byeeeeeee")
        rgb_pil_image = pil_image
        upscaled_image = pipeline(rgb_pil_image, num_inference_steps=params["num_inference_steps"], eta=params["eta"]).images[0]
        if params["image_save"]:
            im_name,_ = os.path.splitext(image_name)        
            save_path = os.path.join("./data/outputs", f"{im_name}_super_resolution.png")                 
            upscaled_image.save(save_path)
        return np.array(upscaled_image)

    def _aligner(self, image_array, image_name, **params):
        
        cascade_fn = "./haarcascade_frontalface_alt2.xml"
        scale=1
        scaleFactor=1.3
        tracker = FaceTracker(cascade_fn,scale,scaleFactor)
        npoints,rects,angle= tracker.detect(image_array)
        face_angle=tracker.face_angle(image_array,npoints)

        
        #=================================#
        # #set right angle to rotate a card#
        #=================================#
        if angle == 90 or angle == 180:
            angle_card = -angle

        elif angle == 270 and face_angle == 90:
            angle_card = -angle

        elif angle == 270 and face_angle < 0:
            angle_card = -angle

        elif angle == 330 and face_angle>0:
            angle_card = 0

        elif angle == 330 and face_angle<0:
            angle_card = -180

        elif angle == 150 and face_angle < 0:
            angle_card = -270

        elif angle == 270 and face_angle != 90:
            face_angle = 90 -face_angle
            angle_card = -(angle+face_angle)

        elif angle==120:
            angle_card = 220

        elif angle==300 and face_angle != 90:
            angle_card = 90		

        else:
            angle_card = face_angle	

        #======================================#
        # rotated cards according to card angle#
        #======================================#

        height, width = image_array.shape[:2] # image shape has 3 dimensions
        image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle_card, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0]) 
        abs_sin = abs(rotation_mat[0,1])
            
        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        
        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(image_array, rotation_mat, (bound_w, bound_h))


        return rotated_mat 
        
    def _bg_remover(self, image_array, image_name, **params):
        
        array_shape = image_array.shape
        # print(array_shape)
        removing_face_part = image_array[int(array_shape[1]* 1/9):, int(array_shape[1]/2):int(array_shape[1]* 5/6) , :]
        pil_image = Image.fromarray(removing_face_part)
        remover = Remover(device= "cpu") 
        blackAndWhiteImage =np.array(remover.process(pil_image, type='white'))

        return blackAndWhiteImage
    

    def _saturation(self, image_array, image_name, **params):
        im_hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
        saturation_factor = 1.5
        im_hsv[:, :, 1] = np.clip(im_hsv[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)

        # Convert back to BGR
        image_saturated_bgr = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        if params["image_save"]:
            im_name,_ = os.path.splitext(image_name)        
            save_path = os.path.join("./data/outputs", f"{im_name}_enlarged_image.png")                 
            self.save_image(image_saturated_bgr, save_path)
        
        return image_saturated_bgr

    def _eadg_smoother(self, image_array, image_name, **params ):
        kernel = np.ones((params["kernel_size"], params["kernel_size"]), np.uint8)
        dilated_image = cv2.dilate(image_array, kernel, iterations=1)
        return dilated_image
    
    def _apply_lowpass_filter(self, image_array, image_name, **params ):
        # Convert binary image to float32
        binary_image_float = image_array.astype(np.float32)

        # Apply Gaussian filter
        smoothed_image_float = cv2.GaussianBlur(binary_image_float, (0, 0), params["sigma"])

        # Convert back to uint8
        smoothed_image = np.round(smoothed_image_float).astype(np.uint8)

        return smoothed_image

    def _resize(self, image_array, image_name, **params ):
        resized_img_array = cv2.resize(image_array.astype(np.uint8), (320, 48), interpolation=cv2.INTER_AREA)
        return resized_img_array

    def show_image(self,image_array, title='Image'):
        cv2.imshow(title, image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self,image_array, output_path):
        cv2.imwrite(output_path, image_array)
        # print(output_path)

    
