function rotated_img = alignImage(img)
    % alignImage uses regionprops and imrotate to rotate embryo images
    % such that their major axis is rotated onto the x axis.
    % Input: image array 
    % Output: rotated image with whole embryo aligned to x axis
    %
    % Written by V. Harihar -- Feb 2, 2024
    %---------------------------------------------------------------------

    %threshold embryo
    thresh = graythresh(img);
    img_gray = double(img)/255 > thresh;
    
    %label embryo region
    se = strel('disk',5,0);
    img_dilated = imdilate(img_gray,se);
    img_labeled = bwlabel(img_dilated);
    
    %obtain orientation angle with regionprops
    props = regionprops(img_labeled, 'Orientation');
    orientation = props.Orientation;

    %rotate image and smooth background
    rotated_img = imrotate(img, orientation, 'bilinear');
    rotated_img(rotated_img < min(min(img))) = img(1,1);
end