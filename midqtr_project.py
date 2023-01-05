"""
DSC 20 Mid-Quarter Project
Name: Sujay Talanki, Daniel Milton
PID: A16433981, A15599774
"""
    
# Part 1: RGB Image #
class RGBImage:
    """
    This class will enable us to initialize new instances of images in \
    RGB color spaces and set pixels to new positions.
    """

    def __init__(self, pixels):
        """
        This method will initialize new instances of a RGBImage and 
        its variables.
        """
        self.pixels = pixels

    def size(self):
        """
        A getter method that returns the size of the image. Since the 
        matrix has 3 dimensions, the length of each sublist will tell us how 
        many rows there are, so we used the len() function on the first sublist
        (all sublists are the same size, so we could realistically index 
        0, 1, or 2 and get the same result). We found the columns by using the 
        len() function on sublists of the sublists (can use 0, 1, or 2).
        """
        self.rows = len(self.pixels[0])
        self.columns = len(self.pixels[0][0])
        return tuple((self.rows, self.columns))

    def get_pixels(self):
        """
        A getter method that returns a deep copy of the pixels of the 
        image. This nested list comprehension creates a new 3d matrix 
        with the exact same intensity values.
        """
        copy_of_pixels = [[[val for val in row] for row in channel] for \
        channel in self.pixels]
        return copy_of_pixels

    def copy(self):
        """
        This method returns a copy of the RGBImage instance. The argument 
        needs the get_pixels() method from the last part in order to create 
        a duplicate copy.
        """
        RGB_instance = RGBImage(self.get_pixels())
        return RGB_instance

    def get_pixel(self, row, col):
        """
        A getter method that returns the color of a pixel at a 
        certain position. This returns a 3-element tuple. Exceptions were 
        raised in accordance to with the write-up.
        """
        blue_channel = 2
        if type(row) != int or type(col) != int:
            raise TypeError()
        elif row >= self.size()[0] or col >= self.size()[1]:
            raise ValueError()
        else:
            red_intensity = self.pixels[0][row][col]
            green_intensity = self.pixels[1][row][col]
            blue_intensity = self.pixels[blue_channel][row][col]
            return tuple((red_intensity, green_intensity, blue_intensity))
        
    def set_pixel(self, row, col, new_color):
        """
        A setter method that updates the color of the pixel at a 
        certain position. Exceptions were raised in accordance to with 
        the write-up. 
        """
        if type(row) != int or type(col) != int:
            raise TypeError()
        elif row >= self.size()[0] or col >= self.size()[1]:
            raise ValueError()
        else:
            for intensity in range(len(new_color)):
                if new_color[intensity] != -1:
                    self.pixels[intensity][row][col] = new_color[intensity]

# Part 2: Image Processing Methods #
class ImageProcessing:
    """
    This class enables us to process and modify images using various methods.
    """

    @staticmethod
    def negate(image):
        """
        This method will return the negated version of a given image. 
        To accomplish this operation, we used the same nested list 
        comprehension algorithm as before, except we subtracted 
        each each intensity value from 255.
        """
        inverted_constant = 255
        inverted_pixel_matrix = [[[inverted_constant - val for val in \
        row] for row in channel] for channel in image.pixels]
        return RGBImage(inverted_pixel_matrix)

    @staticmethod
    def tint(image, color):
        """
        This method takes in a color tuple, and tints the image using the 
        operations specified in the write-up. The nested list comprehension
        algorithmwas used again, but the now the final value appended to the 
        matrix will undergoe the operations specified in the write-up.
        """
        average_divisor = 2
        tinted_matrix = [[[(val + color[channel])//average_divisor for val \
        in image.pixels[channel][row]] for row in \
        range(len(image.pixels[channel]))] for channel in \
        range(len(image.pixels))]
        return RGBImage(tinted_matrix)

    @staticmethod
    def clear_channel(image, channel):
        """
        This method clears the given channel of an image. This will update 
        the intensity value of given channel to 0.
        """
        clear_matrix = [[[0 if channel == chan else val for val \
        in image.pixels[chan][row]] for row in \
        range(len(image.pixels[chan]))] for chan in \
        range(len(image.pixels))]
        return RGBImage(clear_matrix)

    @staticmethod
    def crop(image, tl_row, tl_col, target_size):
        """
        This method will crop an image according the given parameters. The 
        nested list comprehension algorithmwas used again, but list indexing 
        was used to attain only the pixels that were in the valid 
        dimensions. 
        """
        if tl_row + target_size[0] > image.size()[0]:
            end_row = image.size()[0]
        else:
            end_row = tl_row + target_size[0]
        if tl_col + target_size[1] > image.size()[1]:
            end_col = image.size()[1]
        else:
            end_col = tl_col + target_size[1]
        
        cropped_matrix = [[image.pixels[channel][row][tl_col:end_col] for row \
        in range(tl_row, end_row)] for channel in range(len(image.pixels))]
        return RGBImage(cropped_matrix)
        
    @staticmethod
    def chroma_key(chroma_image, background_image, color):
        """
        This method performs a chroma key algorithm with the chroma_image
        and replaces a specified pixel with the same pixel from 
        the background image.
        """
        if type(chroma_image) != RGBImage or \
        type(background_image) != RGBImage:
            raise TypeError()
        elif chroma_image.size() != background_image.size():
            raise ValueError()
        else:
            chroma_matrix = [[[background_image.pixels[channel][row][val] if \
            color == chroma_image.get_pixel(row, val) else \
            chroma_image.pixels[channel][row][val] for val in \
            range(len(chroma_image.pixels[channel][row]))] for row in \
            range(len(chroma_image.pixels[channel]))] for channel in \
            range(len(chroma_image.pixels))]
            return RGBImage(chroma_matrix)

    # rotate_180 IS FOR EXTRA CREDIT (points undetermined)
    @staticmethod
    def rotate_180(image):
        """
        This method will rotate an image 180 degrees by the algorith below.
        A lambda function was created to reverse the elements inside each 
        sublist as well as reverse the order of each sublist. Other than that 
        the same nested list comprehension algorithm was used.
        """
        copy_of_image = image.copy()
        reverse = lambda lst: lst[::-1]
        rotated_pixel_matrix = [list(reversed(list(map(reverse, inner_lst)))) \
        for inner_lst in copy_of_image.pixels]
        return RGBImage(rotated_pixel_matrix)

# Part 3: Image KNN Classifier #
class ImageKNNClassifier:
    """
    This class allows us to predict the label of an image based on the labels
    of the closest images.
    """

    def __init__(self, n_neighbors):
        """
        The constructor initializes the following attributes below.
        """
        self.n_neighbors = n_neighbors
        self.training_data = []
        self.data = []

    def fit(self, data):
        """
        This method 'fits' the training data given in the data parameter into 
        the object's training data. Exceptions are raised in accordance to the
        write-up.
        """
        if len(data) <= self.n_neighbors:
            raise ValueError()
        elif len(self.data) != 0:
            raise ValueError()
        else:
            self.data = data
            for tup in self.data:
                self.training_data.append(tup)

    @staticmethod
    def distance(image1, image2):
        """
        This method flattens the 3d matrices of each image, and uses 
        the euclidean distance formula to calculate the distance between 
        the images. Exceptions were raised in accordance with the 
        write-up.
        """
        squared = 2
        square_root = .5
        
        if type(image1) != RGBImage or type(image2) != RGBImage:
            raise TypeError()
        elif image1.size() != image2.size():
            raise ValueError()
        else: 
            flattened_image1 = [image1.pixels[channel][row][val] for channel \
            in range(len(image1.pixels)) for row in \
            range(len(image1.pixels[channel])) for val in \
            range(len(image1.pixels[channel][row]))]
        
            flattened_image2 = [image2.pixels[channel][row][val] for channel \
            in range(len(image2.pixels)) for row in \
            range(len(image2.pixels[channel])) for val in \
            range(len(image2.pixels[channel][row]))]
        
            differences = [flattened_image2 - flattened_image1 for \
            flattened_image2, flattened_image1 in zip(flattened_image2, \
            flattened_image1)]
            discriminant = sum([value**squared for value in differences])
            distance = discriminant**square_root
            return distance

    @staticmethod
    def vote(candidates):
        """
        This method finds the most popular label among the elements by 
        finding the maximum number of times a label occured. 
        """
        count = 0
        no = candidates[0]
        for label in candidates:
            current_freq = candidates.count(label)
            if (current_freq >= count):
                count = current_freq
                most_freq = label
        return most_freq

    def predict(self, image):
        """
        This function uses the distance method to find the distances between 
        the images in the training data and the given image parameter. 
        This list is sorted and only the k closest neighbors are kept. Lastly, 
        the vote method is used to find the most popular label in this list, 
        and makes a predction of the label based on the information.
        """
        if len(self.data) == 0:
            raise ValueError()
        else:
            sorted_data = sorted([tuple((ImageKNNClassifier.distance(image, \
            tup[0]), tup)) for tup in self.data])
            neighbors = sorted_data[:self.n_neighbors]
            labels_list = [neighbor[1][1] for neighbor in neighbors]
            prediction = ImageKNNClassifier.vote(labels_list)
            return prediction
