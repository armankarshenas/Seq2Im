function [BW,maskedImage] = segmentImage(X)
%segmentImage Segment image using auto-generated code from imageSegmenter app
%  [BW,MASKEDIMAGE] = segmentImage(X) segments image X using auto-generated
%  code from the imageSegmenter app. The final segmentation is returned in
%  BW, and a masked image is returned in MASKEDIMAGE.

% Auto-generated by imageSegmenter app on 01-Feb-2024
%----------------------------------------------------


% Adjust data to span data range.
X = imadjust(X);

% Threshold image - global threshold
BW = imbinarize(X);

% Erode mask with disk
radius = 2;
decomposition = 8;
se = strel('disk', radius, decomposition);
BW = imerode(BW, se);

% Create masked image.
maskedImage = X;
maskedImage(~BW) = 0;

% Create the structure for the embryos 
labeledImage = bwlabel(BW);
embryos = regionprops(labeledImage, 'BoundingBox');

% Applying area restriction on the bounding boxes
coordinates = [];
for i = 1:numel(embryos)
    area = embryos(i).BoundingBox(3)*embryos(i).BoundingBox(4);
    if area > 150000 && area <250000
        coordinates = [coordinates;embryos(i).BoundingBox];
    end
end
end
