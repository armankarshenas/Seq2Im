%% *Embryo Segmentation Project Trials*
% 
% 
% _Notes from Yuxin:_
% 
% _This is a documentation/supplementary codes to demonstrate the sequence of 
% the steps/origin of the values used in the final version of the Segmentation 
% function._ 
% 
% _Here, I used the image *"013619~A-M-pyramid.tiff"* as an example_ 
% 
% 
%% *Step 1: upload image + downsize* 
%% 
% * downsizing is unnecessary if computer's capacity permits high performance.

emb_im2 = imread("013619~A-M-pyramid.tiff");
emb_im2_ds = imresize(emb_im2,0.7,'nearest');
%% *Step 2: enchance image contrast*

emb2_adj = imadjust(emb_im2_ds);
%% *Step 3: Binary Construction*
%% 
% * generate pixel intensity historgam
% * visually inspect histogram to manually pick a threshold estimate

imhist(emb_adj)
title('Intensity Histogram of Embryo Image')
ylabel('Frequency')
xlabel('Pixel Value')
%% 
% * Convert the image into binary from manual thresholding (intensity = 200)
% * Invert the binary (0-->1; 1-->0) so that "imfill" works
% * Fill holes within connected objects
% * Remove impurities by manual selection of pixel area =< 6000, connectivity 
% = 4 for better removability 

BW2 = emb_adj > 200; 
BW2_inverted = imcomplement(BW2);
BW2_filled = imfill(BW2_inverted, "holes");
BW2_clear = bwareaopen(BW2_filled, 6000,4);
%% *Step 4: First Method of Area Selection _(USED for initial setup)*_
%% 
% * Segment every object in the clear image by bounding box (draw a rectangle 
% around the object) and retrieve area properties

CC0 = bwconncomp(BW2_clear);
stats0 = regionprops("table",CC0,"Area","BoundingBox");
%% 
% * generate a histogram: visualize areas of each object cut out from bounding 
% box

area0 = stats0.Area;
histogram(area0,size(area0,1),'Normalization', 'percentage');
xlabel('Area of Segmented Object (pixel size)');
ylabel('Frequency in Percentage');
title('Distribution of the Pixel Areas of the Segmented Objects');
%% 
% * Another visualization: boxplot

figure;
boxplot(area0);
xlabel('All Areas of Bounding Objects');
ylabel('Areas in Pixels');
title('Bounding Area Distribution Visualization');
%% 
% * determines the area range (middle 75% of all bounded object areas)

area_range = quantile(area0,[0.125 0.875])
%% 
% * generate the clean binary image with connected embryos eliminated using 
% area restriction 

emb2_selection = (area0 > area_range(1)) & (area0 < area_range(2));
BW2_clean = cc2bw(CC0,ObjectsToKeep=emb2_selection);
%% *Step 5: Label embryos*
%% 
% * retrieve max number of labels

label0 = bwlabel(BW2_clean,4);
max_labels0 = max(label0, [],"all")
%% 
% * Find mean and standard deviation of 


%% Final Step - Segmented Image Storage
%% 
% * Create a folder to store segmented images

output_folder = 'segmented_embryos2/';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
%% 
% * Loop through labeled images, crop them using bounding box, convert back 
% to original, and store in folder, each image named an embryo ID #

num_images = max_labels0;

for l = 1:num_images 

    stats2 = regionprops(label0==l, 'BoundingBox', 'Area'); 
 
    boundingBox = stats2(1).BoundingBox;
    Cropped_image = imcrop(emb_im2_ds, boundingBox);
    filename = sprintf('2embryo_%d.jpg', l);
    full_path = fullfile(output_folder, filename);
   
    imwrite(Cropped_image, full_path);
end
%% -----------------Above are initial setup--------------------
%% *Relative Threshold*
% To ensure more accurate thresholding, we want to calculate the relative threshold
%% 
% * invert the embryos BW & map them onto the original image so that the latter 
% only contains the background while all embryos are shown as black
% * Create a background-only image

BW2_clean_opp = imcomplement(BW2_clean);
bkg = emb_im2_ds .* uint8(BW2_clean_opp);
%% 
% * create a mask to store zeros in bkg + count the numbes of zeros
% * Count the number of zeros
% * find the mean intensity value of the background (disregard zeros)
%% 
% _limitation of this step: black lines of the original image at the edges are 
% excluded from background intensity calculation_

zero_mask = (bkg == 0);
num_zeros = sum(zero_mask(:))
bkg_size = size(bkg,1)*size(bkg,2)
mean_bkg_intensity = sum(bkg(:))/(bkg_size - num_zeros)
%% 
% * Find the mean intensity of the segmented embryos using regionprops

stats_0 = regionprops(BW2_clean,emb_im2_ds,"MeanIntensity");
emb_mean_intensities = [stats_0.MeanIntensity];
mean_emb_intensity = sum(emb_mean_intensities) / size(emb_mean_intensities,2)
%% 
% * Calculate threshold percent (e.g. this means that avg background intensity 
% is about 17.95% lighter than embryo intensity)
%% 
% * Calculate relative threshold value
% * _*k*_ is preselected, and can be adjusted as needed to select threshold 
% performance
% * relative threshold ~ *226* 

threshold_percent = (mean_bkg_intensity - mean_emb_intensity)/ mean_bkg_intensity
k = 0.5
rel_threshold = k*(threshold_percent*mean_bkg_intensity)+mean_emb_intensity
%% 
% * Visualize the two thresholds

imhist(emb_adj)
title('Intensity Histogram of Image 2')
ylabel('Frequency')
xlabel('Pixel Value')
hold on
xline(200, 'r', 'LineWidth', 2)
xline(226, 'b', 'LineWidth', 2)
hold off
%% Mean Embryo Area
%% 
% * Import 7 selected embryo images (from previous segmentation)
% * These are manually + randomly chosen to represent different stages of embryo 
% development 
% * link to these images: <https://drive.google.com/drive/folders/1ciesk336zBUoZcbTI8jjJmLaGANWoTl-?usp=sharing 
% https://drive.google.com/drive/folders/1ciesk336zBUoZcbTI8jjJmLaGANWoTl-?usp=sharing>

emb1 = imread("embryo_52.jpg");
emb2 = imread("embryo_43.jpg");
emb3 = imread("embryo_28.jpg");
emb4 = imread("embryo_9.jpg");
emb5 = imread("embryo_17.jpg");
emb6 = imread("embryo_25.jpg");
emb7 = imread("embryo_33.jpg");
%% 
% * Remove backgrounds (using relative threshold = 226), fill the holes, remove 
% impurities (72 = picture DPI) & store segmented embryos in a cell
% * P.S. "+1" because of the header in the first row 

emb_imgs = {emb1,emb2,emb3,emb4,emb5, emb6,emb7};
num_emb = length(emb_imgs);
emb_store = [{'emb_masks','emb_filled_masks','emb_clear_masks','emb_nobkg'};cell(num_emb,4)];

for i = 1:num_emb
    emb_store{1+i,1} = emb_imgs{i} < 226; 
    emb_store{1+i,2} = imfill(emb_store{1+i,1}, "holes");
    emb_store{1+i,3} = bwareaopen(emb_store{i+1,2},72,4);
    emb_store{1+i,4} = emb_imgs{i} .* uint8(emb_store{1+i,2});
end

emb_store
%% 
% * Calculate each embryo area & find mean embryo area -> area restriction

emb_areas = cell(num_emb, 1);

for i = 1:num_emb
    emb_areas{i} = regionprops(emb_store{1+i,3}, 'area','BoundingBox').Area;
end
emb_areas

mean_emb_area = sum(cell2mat(emb_areas))/num_emb
%% New area restriction method
%% 
% * Replace the threshold value in the initial setup to the new one & run through

emb_adj = imadjust(emb_im2_ds); %enhance contrast
BW = emb_adj > 226; %relative threshold
BW_invert = imcomplement(BW); 
BW_filled = imfill(BW_invert, "holes"); %fill "holes" in objects
BW_clear = bwareaopen(BW_filled, 6000,4); %remove impurities manual selection

CC = bwconncomp(BW_clear,4);
stats = regionprops("table",CC,"Area","BoundingBox");
area = stats.Area;
%% 
% * Visualize the new area distribution - histogram

histogram(area,size(area,1),'Normalization', 'percentage');
xlabel('Area of Segmented Object (pixel size)');
ylabel('Frequency in Percentage');
title('Distribution of the Pixel Areas of the Segmented Objects');
%% 
% * Using mean embryo area = *63805* px & k = 0.07 (adjust as needed) for new 
% area range
% * Determine the outlier boundaries (lower_bound & upper_bound) from IQR
% * Generate a boxplot that represents the area range of segmented objects (with 
% outliers filtered; mean embryo area & its lower and upper bound shown)

Q1 = prctile(area, 25);
Q3 = prctile(area, 75);
IQR = Q3 - Q1;

lower_bound = Q1 - 1.5 * IQR;
upper_bound = Q3 + 1.5 * IQR;

k = 0.07;
area_Lbound = mean_emb_area*(1-k)
area_Ubound = mean_emb_area*(1+k)


filtered_area = area(area >= lower_bound & area <= upper_bound);

figure;
boxplot(filtered_area, 'Labels', {'Filtered area'});
title('Filtered Bounding Area');
hold on
plot(mean_emb_area, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); 
hold on
plot(area_Ubound, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
plot(area_Lbound, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'g');

hold off
%% 
% * Use new area range to eliminate connected embryos
% * Generate new labels 

area_range = [area_Lbound, area_Ubound]
emb_selection = (area > area_range(1)) & (area < area_range(2));
BW_clean = cc2bw(CC,ObjectsToKeep=emb_selection);
label = bwlabel(BW_clean,4);  
max_labels = max(label, [],"all")
%% 
% * Generate new folder, example ID = 5
% * Loop through labeled images, crop them using bounding box, convert back 
% to original, and store in folder, each image named an embryo ID #

output_folder = sprintf('segmented_embryos_%d/', 5);
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    for i = 1:max_labels
        stats2 = regionprops(label==i, 'BoundingBox'); 
        boundingBox = stats2(1).BoundingBox; 
        segmentedImage = imcrop(emb_im2_ds, boundingBox);
        filename = sprintf('embryo_%d.jpg', i);
        full_path = fullfile(output_folder, filename);
        imwrite(segmentedImage, full_path); 
    end
%% Final step: Put everything into a callable function
%% 
% * See the "embseg_func.m" file
%% *Extra considerations/improvements to make:*
%% 
% * apply low-pass filter to remove initial noise of the image
% * select a more appropriate k as needed (**note that there are _*two k's,*_ 
% one is used for relative threshold, one is used for segment more embryos per 
% image (inside function))
% * figure out how to calculate the segmentation efficiency
% * _*Potential caveat!*_ If choose to not downsize the image, then the mean 
% embryo area I've calculated above would not be accurate. Above only displays 
% mean area when downsizing factor = 0.7. Therefore, mean embryo area of the original 
% resolution would be > ~ 6E04 px.
% * Another issue: For embryos that are very close to one another though not 
% connected can still be seen in the segmented images (e.g. in the corner)
%% 
%