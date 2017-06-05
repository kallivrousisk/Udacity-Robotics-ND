import numpy as np
import cv2
#from math import sqrt

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# finding obstacles = not path = inverse of path
# R and G and B need to be bellow threshold

def color_thresh_obst(img, rgb_thresh=(90, 90, 90)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    below_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[below_thresh] = 1
    # Return the binary image
    return color_select
 

def color_thresh_rock(img, rgb_thresh=(20, 100, 100)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    rock_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[rock_thresh] = 1
    # Return the binary image
    return color_select


def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(x_pixel, y_pixel, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    # Apply a rotation
    x_pixel_rotated = (x_pixel * np.cos(yaw_rad)) - (y_pixel * np.sin(yaw_rad))
    y_pixel_rotated = (x_pixel * np.sin(yaw_rad)) + (y_pixel * np.cos(yaw_rad))
    # Return the result  
    return x_pixel_rotated, y_pixel_rotated


# Define a function to perform a translation
def translate_pix(x_pixel_rotated, y_pixel_rotated, xpos, ypos, scale):
    # Apply a scaling and a translation
    x_pixel_translated = (x_pixel_rotated / scale) + xpos
    y_pixel_translated = (y_pixel_rotated / scale) + ypos
    # Return the result
    return x_pixel_translated, y_pixel_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    
    # important to remmebert where the image is now actually coming from!!!!!!!!!!
    img = Rover.img
    dst_size = 5 
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset]])
    
    # 2) Apply perspective transform
    warped = perspect_transform(img, source, destination)
    
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed_path = color_thresh(warped, rgb_thresh=(160, 160, 160))
    threshed_obst = color_thresh_obst(warped, rgb_thresh=(160, 160, 160))
    threshed_rock = color_thresh_rock(warped, rgb_thresh=(20, 100, 100))
    
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
        
    Rover.vision_image[:, :, 0] = threshed_obst * 255
    Rover.vision_image[:, :, 1] = threshed_rock * 255
    Rover.vision_image[:, :, 2] = threshed_path * 255   
        

    # 5) Convert map image pixel values to rover-centric coords
    
    path_xpix, path_ypix = rover_coords(threshed_path)
    obstacles_xpix, obstacles_ypix = rover_coords(threshed_obst)
    rocks_xpix, rocks_ypix = rover_coords(threshed_rock)
    
    
    
    # 6) Convert rover-centric pixel values to world coordinates
    
    xpos, ypos, yaw = Rover.pos[0], Rover.pos[1], Rover.yaw
    world_size = Rover.worldmap.shape[0]
    scale = 10
    path_x_world, path_y_world = pix_to_world(path_xpix, path_ypix, xpos, ypos, yaw, world_size, scale)
    obstacles_x_world, obstacles_y_world = pix_to_world(obstacles_xpix, obstacles_ypix, xpos, ypos, yaw, world_size, scale)
    rocks_x_world, rocks_y_world = pix_to_world(rocks_xpix, rocks_ypix, xpos, ypos, yaw, world_size, scale)  
    
    
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
        
    if Rover.roll < 2.0 or Rover.roll > 358:
        if Rover.pitch < 2.0 or Rover.pitch > 358:
            Rover.worldmap[path_y_world, path_x_world, 2] += 255
            Rover.worldmap[rocks_y_world, rocks_x_world, 1] += 255
            Rover.worldmap[obstacles_y_world, obstacles_x_world, 0] += 255


    
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(path_xpix, path_ypix)
    Rover.rocks_dists, Rover.rocks_angles = to_polar_coords(rocks_xpix, rocks_ypix)
    Rover.obstcls_dists, Rover.obstcls_angles = to_polar_coords(obstacles_xpix, obstacles_ypix)
    
     
    return Rover