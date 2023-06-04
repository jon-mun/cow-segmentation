import numpy as np
import matplotlib.pyplot as plt
import math

class I:
    '''
    Custom Image Class
    '''
    image=[[]]
    
    def __init__(self, image: list):
        self.image = image
        
    @staticmethod
    def from_np(image: np.ndarray) -> 'I':
        # convert np array to list
        image = image.tolist()
        return I(image)
    
    @staticmethod
    def read(img_path) -> list:
        img = plt.imread(img_path)
        img.tolist()
        return img
    
    @staticmethod
    def show(img: list, gray=False) -> None:
        params = {}
        if gray:
            params['cmap'] = 'gray'
        
        plt.imshow(img, **params)
        plt.show()
    
    def display(self) -> None:
        plt.imshow(self.image)
        plt.show()
        
    @staticmethod
    def to_gray(img: list) -> list:
        gray_img = []
        for row in img:
            gray_row = []
            for px in row:
                gray_px = int(I.rgb_to_gray_px(px[0], px[1], px[2]))
                gray_row.append(gray_px)
            gray_img.append(gray_row)
        return gray_img
    
    def rgb_to_gray_px(r, g, b):
        return (r + g + b) / 3
        
    @staticmethod
    def flatten(img: list) -> list:
        '''
        flatten a 3D image to 2D
        '''
        flattened_img = []
        for row in img:
            for px in row:
                flattened_img.append(px)
        return flattened_img
        
    @staticmethod
    def extract_channels(img:list) -> list:
        '''
        extract all channels from an image
        '''
        channels = []
        for i in range(3):
            channels.append(I.extract_channel(img, i))
        return channels
    
    @staticmethod
    def extract_channel(img: list, channel: int) -> list:
        '''
        extract a channel from an image
        '''
        extracted_img = []
        for row in img:
            extracted_row = []
            for px in row:
                extracted_px = px[channel]
                extracted_row.append(extracted_px)
            extracted_img.append(extracted_row)
        return extracted_img
        
    @staticmethod
    def bl_resize(original_img, new_h, new_w):
        src_h, src_w, c = I.shape(original_img)
        resized_img = []

        # Calculate scaling factors
        scale_h = float(src_h) / new_h
        scale_w = float(src_w) / new_w

        # Iterate over each pixel in the resized image
        for y in range(new_h):
            row = []
            for x in range(new_w):
                # Calculate the corresponding position in the original image
                src_y = (y + 0.5) * scale_h - 0.5
                src_x = (x + 0.5) * scale_w - 0.5

                # Get the four surrounding pixels
                src_y1 = int(math.floor(src_y))
                src_x1 = int(math.floor(src_x))
                src_y2 = min(src_y1 + 1, src_h - 1)
                src_x2 = min(src_x1 + 1, src_w - 1)

                # Calculate the weights for interpolation
                w1 = (src_y2 - src_y) * (src_x2 - src_x)
                w2 = (src_y2 - src_y) * (src_x - src_x1)
                w3 = (src_y - src_y1) * (src_x2 - src_x)
                w4 = (src_y - src_y1) * (src_x - src_x1)

                # Perform bilinear interpolation for each color channel
                interpolated_px = []
                for ch in range(c):
                    interpolated_ch = (
                        original_img[src_y1][src_x1][ch] * w1 +
                        original_img[src_y1][src_x2][ch] * w2 +
                        original_img[src_y2][src_x1][ch] * w3 +
                        original_img[src_y2][src_x2][ch] * w4
                    )
                    interpolated_px.append(int(interpolated_ch))
                row.append(interpolated_px)
            resized_img.append(row)

        return resized_img
    
    @staticmethod
    def rgb_to_hsv(rgb_img):
        hsv_img = []
        for row in rgb_img:
            hsv_row = []
            for px in row:
                hsv_px = I.rgb_to_hsv_px(px[0], px[1], px[2])
                hsv_row.append(list(hsv_px))
            hsv_img.append(hsv_row)
        return hsv_img

    def rgb_to_hsv_px(r, g, b):
        r, g, b = r/255.0, g/255.0, b/255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx-mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g-b)/df) + 360) % 360
        elif mx == g:
            h = (60 * ((b-r)/df) + 120) % 360
        elif mx == b:
            h = (60 * ((r-g)/df) + 240) % 360
        if mx == 0:
            s = 0
        else:
            s = (df/mx)*100
        v = mx*100
        return h, s, v
    
    @staticmethod
    def shape(img_list: list):
        row = len(img_list)
        col = len(img_list[0])
        c = len(img_list[0][0])
        return row, col, c
    
    @staticmethod
    def multiply_img(k: float, img):
        '''
        multiply each pixel in an image by a constant
        '''
        new_img = []
        
        for row in img:
            new_row = []
            for px in row:
                if type(px) == list:
                    new_px = [k * channel for channel in px]
                else:
                    new_px = k * px
                new_row.append(new_px)
            new_img.append(new_row)
            
        return new_img
    
    @staticmethod 
    def normalize(img, new_min=0, new_max=1):
        '''
        input: binary image
        normalize an image to a new range
        '''
        img_min = min(img)
        img_max = max(img)
        
        new_img = []
        
        for row in img:
            new_row = []
            for px in row:
                print(type(px))
                new_px = (px - img_min) * (new_max - new_min) / (img_max - img_min) + new_min
                new_row.append(new_px)
            new_img.append(new_row)
            
        return new_img            