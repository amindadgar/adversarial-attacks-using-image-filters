from feature_squeeze import reduce_precision_np, median_filter_np
import numpy as np
from filters.filters import img_filters
from tensorflow import keras


class fitness:
    def __init__(self, tf_model, filters_functions):
        """
        Detection rate fitness
        """
        self.tf_model = tf_model
        self.filters_functions = filters_functions
        self.images_original = None
        self._initialize_images()
        

    def _initialize_images(self):
        """
        initialize the images needed for the problem
        """
        (x_train, _), (_, _) = keras.datasets.cifar10.load_data()

        ## working on portion of data because of low available resources
        portion = 20
        self.images_original = x_train[:portion]
        
        self.original_images_predictions = self.tf_model.predict(self.images_original, verbose=0)

    def _attack_detection(self, images, threshold=1.7547):
        image_squeezed = reduce_precision_np(images, npp=5)
        image_locally_smoothed = median_filter_np(images, width=4)


        # prediction_image = self.tf_model.predict(images, verbose=0) 
        prediction_image = self.original_images_predictions 
        prediction1 = self.tf_model.predict(image_squeezed, verbose=0)
        prediction2 = self.tf_model.predict(image_locally_smoothed, verbose=0)

        distance1 = np.sum(np.abs(prediction_image - prediction1), axis=1)
        distance2 = np.sum(np.abs(prediction_image - prediction2), axis=1)

        ## True will show that the attack was successful and False unsuccessful 
        return (distance1 >= threshold) | (distance2 >= threshold)

    def _evaluate_dr(self, images_batch):
        """
        evaluate the detection rate

        Parameters:
        ------------
        images_batch : numpy array
            the dimension should be (count of images, width=32, height=32, img_channel=3)
            normally the filtered image should be as input here

        Returns:
        ---------
        detection_rate : float
            showing how the model could detect the adversarial attacks
        """
        ## if just one image was as input
        if images_batch.shape[0] == 1:
            images = np.expand_dims(images_batch, axis=1)
        else:
            images = images_batch

        attacks = self._attack_detection(images)

        ## we can get the adversarial images using the line below
        # adversarial_images = images_batch[attacks]

        detection_rate = (1 / len(attacks)) * np.sum(attacks)
        
        return detection_rate
    
    def _evaluate_asr(self, images_original, images_filtered):
        """
        evaluate the attack success rate
        the bigger the value the more success the attacks are

        Parameters:
        ------------
        images_original : numpy array
            the dimension should be (count of images, width=32, height=32, img_channel=3)
        images_filtered : numpy array
            original images that some pre-processing happened to them to attack the model
            the dimension should be (count of images, width=32, height=32, img_channel=3)
            

        Returns:
        ---------
        detection_rate : float
            showing how the model could detect the adversarial attacks
        """
        ## if they were just one image
        ## change the dimension to be compatible with model's input shape
        if images_original.shape[0] == 1:
            images_original_ = np.expand_dims(images_original, axis=0)
        else:
            images_original_ = images_original
        
        if images_filtered.shape[0] == 1:
            images_filtered_ = np.expand_dims(images_filtered, axis=0)
        else:
            images_filtered_ = images_filtered


        # predictions_original = self.tf_model.predict(images_original_, verbose=0)
        predictions_original = self.original_images_predictions
        predictions_filtered = self.tf_model.predict(images_filtered_, verbose=0)

        ## the class with maximum probability
        predictions_classes_original = np.argmax(predictions_original, axis=1)
        predictions_classes_filtered = np.argmax(predictions_filtered, axis=1)

        ## compare the predictions
        same_prediction_count = np.sum(predictions_classes_original != predictions_classes_filtered)

        ## attack success rate
        attack_sr = (1 / len(images_original_)) * same_prediction_count
        
        return attack_sr
        
    
    def _apply_multiple_filters(self, filter_functions, images_batch):
        """
        apply multiple filters on one image
        filter_functions are the multiple filters function extracted from a chromosome 

        returns the manipulated images (filtered ones)
        """
        ## filtered images are saved in the array
        manipulated_images_arr = []

        ## for images
        for image in images_batch:
            manipulated_image = np.copy(image)

            ## Apply multiple filters to the image
            for filter_func in filter_functions:
                manipulated_image = filter_func.apply(manipulated_image)
            
            manipulated_images_arr.append(manipulated_image)
        ## convert back to numpy array
        manipulated_images_arr = np.array(manipulated_images_arr)
        
        return manipulated_images_arr

    def _apply_filter(self, filter_function, images_array):
        """
        apply filter for multiple images (images_array)
        """
        images_filtered = []
        for image in images_array:
            images_filtered.append(filter_function.apply(image))
        
        images_filtered = np.array(images_filtered)
        return images_filtered    

    def fitness_dr(self):
        """
        get the detection rate of the specified filters parameters in chromosome_filters

        # Parameters:
        # ------------
        # chromosome_filter : array of floats
        #     float array with length equal to 3

        Returns:
        ---------
        detection_rate : float
            showing how the model could detect the adversarial attacks
        """
        
        images_filtered = self._apply_multiple_filters(self.filters_functions, self.images_original)

        detection_rate = self._evaluate_dr(images_filtered)

        return detection_rate
    
    def fitness_asr(self):
        """
        find the attack success rate of the specilized image filter on model
        the bigger the value the more success the attacks are

        # Parameters:
        # -------------
        # chromosome_filters : array of floats
        #     float array with length equal to 3

        Returns:
        ---------
        detection_rate : float
            showing how the model could detect the adversarial attacks
        """
        # images_filtered = self._apply_filter(self.images_original, filter_function)
        images_filtered = self._apply_multiple_filters(self.filters_functions, self.images_original)

        attack_sr = self._evaluate_asr(self.images_original, images_filtered)

        return attack_sr

