%% *Embryo Segmentation Project*
% *Initial setup: upload image* 

%example here

emb_im = imread("013619~A-M-pyramid.tiff"); %upload image
emb_im_ds = imresize(emb_im,0.7,'nearest'); %downsize (optional)
%% 
% *Segmentation Function*

function segmentedImage = embryoSegmentation(inputImage, inputID)
   
    emb_adj = imadjust(inputImage); %enhance contrast
    BW = emb_adj > 226; %relative threshold
    BW_invert = imcomplement(BW); 
    BW_filled = imfill(BW_invert, "holes"); %fill "holes" in objects
    BW_clear = bwareaopen(BW_filled, 6000,4); %remove impurities manual selection
    
    CC = bwconncomp(BW_clear);
    stats = regionprops("table",CC,"Area","BoundingBox");
    area = stats.Area;
    
    mean_emb_area = 63805;
    k = 0.07;
    area_Lbound = mean_emb_area*(1-k);
    area_Ubound = mean_emb_area*(1+k);

    area_range = [area_Lbound, area_Ubound];

    %eliminate overlapping embryos
    emb_selection = (area > area_range(1)) & (area < area_range(2));
    BW_clean = cc2bw(CC,ObjectsToKeep=emb_selection);

    label = bwlabel(BW_clean,4);  %label objects
    max_labels = max(label, [],"all");


    % create a folder to save the segmented images
    output_folder = sprintf('segmented_embryos_%d/', inputID);
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    for i = 1:max_labels %loop through every predetermined label
    
        stats2 = regionprops(label==i, 'BoundingBox'); % store properties of the labeled embryo 
        %extract bounding box of the labeled embryo, index = 1 because only 1 region
        boundingBox = stats2(1).BoundingBox; 
        segmentedImage = imcrop(inputImage, boundingBox);% crop the original image using the bounding box
      
        filename = sprintf('embryo_%d.jpg', i);
        full_path = fullfile(output_folder, filename);
        imwrite(segmentedImage, full_path); % Save the image in the folder
    end

end


%% 
% *Execute the function*

embryoSegmentation(emb_im_ds,6) %use original image as needed