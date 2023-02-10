import numpy as np
import cv2 as cv
import filters.util as util
import filters.css as css
from PIL import Image

class img_filters():

    def __init__(self, alpha, strength, filter_name) -> None:
        """
        initialize the parameters of image filters
        """
        self.alpha = alpha
        self.strength = strength

        filter_functions = {
            'kelvin' : self.kelvin,
            'clarendon' : self.clarendon,
            'moon' : self.moon,
            # 'sharpening': self.sharpening
            'gingham': self.gingham
        }
        self.filtering_function = filter_functions[filter_name]


    def apply(self, image):
        """
        apply the filter on the original image and return it
        """
        filtered_img = self.filtering_function(image)
        ## applying strength
        img_output = self._apply_strength(image, np.array(filtered_img), self.strength)

        ## convert to unsinged integer type 
        ## because applying strenght and alpha to the image may create float values between 0 to 255
        img_output = np.array(img_output, dtype=np.uint8)

        return img_output

    def kelvin(self, image):
        # create a copy of input image to work on
        output = image.copy()

        #split the channels
        blueChannel, greenChannel, redChannel = cv.split(output)

        #Interpolation values
        redValuesOriginal = np.array([0, 60, 110, 150, 235, 255])
        redValues =         np.array([0, 102, 185, 220, 245, 245 ])
        greenValuesOriginal = np.array([0, 68, 105, 190, 255])
        greenValues =         np.array([0, 68, 120, 220, 255 ])
        blueValuesOriginal = np.array([0, 88, 145, 185, 255])
        blueValues =         np.array([0, 12, 140, 212, 255])

        #create lookup table
        allValues = np.arange(0, 256)
        #Creating the lookuptable for blue channel
        blueLookuptable = np.interp(allValues, blueValuesOriginal, self.alpha * blueValues)
        #Creating the lookuptable for green channel
        greenLookuptable = np.interp(allValues, greenValuesOriginal, self.alpha * greenValues)
        #Creating the lookuptable for red channel
        redLookuptable = np.interp(allValues, redValuesOriginal, self.alpha * redValues)

        #Apply the mapping for blue channel
        blueChannel = cv.LUT(blueChannel, blueLookuptable)
        #Apply the mapping for green channel
        greenChannel = cv.LUT(greenChannel, greenLookuptable)
        #Apply the mapping for red channel
        redChannel = cv.LUT(redChannel, redLookuptable)

        #merging back the channels
        output = cv.merge([blueChannel, greenChannel, redChannel])

        #convert to uint8
        output = np.uint8(output)

        return output

    def clarendon(self, image):
        #create a copy of input image to work on
        clarendon = image.copy()

        #split the channels
        blueChannel, greenChannel, redChannel = cv.split(clarendon)

        #Interpolation values
        originalValues = np.array([0, 28, 56, 85, 113, 141, 170, 198, 227, 255])
        blueValues =     np.array([0, 38, 66, 104, 139, 175, 206, 226, 245, 255 ])
        redValues =      np.array([0, 16, 35, 64, 117, 163, 200, 222, 237, 249 ])
        greenValues =    np.array([0, 24, 49, 98, 141, 174, 201, 223, 239, 255 ])

        #Creating the lookuptables
        fullRange = np.arange(0,256)
        #Creating the lookuptable for blue channel
        blueLookupTable = np.interp(fullRange, originalValues, self.alpha * blueValues )
        #Creating the lookuptables for green channel
        greenLookupTable = np.interp(fullRange, originalValues, self.alpha * greenValues )
        #Creating the lookuptables for red channel
        redLookupTable = np.interp(fullRange, originalValues, self.alpha * redValues )

        #Apply the mapping for blue channel
        blueChannel = cv.LUT(blueChannel, blueLookupTable)
        #Apply the mapping for green channel
        greenChannel = cv.LUT(greenChannel, greenLookupTable)
        #Apply the mapping for red channel
        redChannel = cv.LUT(redChannel, redLookupTable)

        #merging back the channels
        output = cv.merge([blueChannel, greenChannel, redChannel])

        #convert to uint8
        output = np.uint8(output)


        return output

    def moon(self, image):
    
        #create a clone of input image to work on
        output = image.copy()

        #convert to LAB color space
        output = cv.cvtColor(output, cv.COLOR_BGR2LAB)

        #split into channels
        L, A, B = cv.split(output)

        #Interpolation values
        originalValues = np.array([0, 15, 30, 50, 70, 90, 120, 160, 180, 210, 255 ])
        values = np.array([0, 0, 5, 15, 60, 110, 150, 190, 210, 230, 255  ])

        #create lookup table
        allValues = np.arange(0, 256)

        #Creating the lookuptable
        lookuptable = np.interp(allValues, originalValues, self.alpha * values)

        #apply mapping for L channels
        L = cv.LUT(L, lookuptable)

        #convert to uint8
        L = np.uint8(L)

        #merge back the channels
        output = cv.merge([L, A, B])

        #convert back to BGR color space
        output = cv.cvtColor(output, cv.COLOR_LAB2BGR)

        #desaturate the image
        #convert to HSV color space
        output = cv.cvtColor(output, cv.COLOR_BGR2HSV)

        #split into channels
        H, S, V = cv.split(output)

        #Multiply S channel by saturation scale value
        S = S * 0.01

        #convert to uint8
        S = np.uint8(S)

        #limit the values between 0 and 256
        S = np.clip(S, 0, 255)

        #merge back the channels
        output = cv.merge([H, S, V])

        #convert back to BGR color space
        output = cv.cvtColor(output, cv.COLOR_HSV2BGR)

        return output

    def gingham(self, image):
        """
        # Copyright 2019 Akiomi Kamakura
        
        Applies Gingham filter.

        Arguments:
            im: An input image.

        Returns:
            The output image.
        """

        image = Image.fromarray(image)

        cb = util.or_convert(image, "RGB")


        cs = util.fill(cb.size, [int(230 * self.alpha), int(230 * self.alpha), int(250 * self.alpha)])

        cr = css.blending.soft_light(cb, cs)

        cr = css.brightness(cr, 1.05)
        cr = css.hue_rotate(cr, -10)

        return cr

    def sharpening(self, image):
        '''
        This function will create the Sharpening filter like effect on an image.
        Args:
            image:  The image on which the filter is to be applied.
            display: A boolean value that is if set to true the function displays the original image,
                        and the output image, and returns nothing.
        Returns:
            output_image: A copy of the input image with the Sharpening filter applied. 
        '''

        # Get the kernel required for the sharpening effect.
        sharpening_kernel = self.alpha * np.array([[-1, -1, -1],
                                        [-1, 9.2, -1],
                                        [-1, -1, -1]])

        # Apply the sharpening filter on the image.
        output_image = cv.filter2D(src=image, ddepth=-1, 
                                    kernel=sharpening_kernel)

        return output_image

    def _apply_strength(self, image, filtered_img, strength):
        """
        Combine the original image and the filtered image with a coefficient named strength
        """
        img_output = (1 - strength) * image + strength * filtered_img 

        return img_output

