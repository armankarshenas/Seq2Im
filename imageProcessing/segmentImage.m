function coordinates = segmentImage(X)

% segmentImage Segment image and return a structure of embryos with bounding 
% boxes around embryos that meet an area restriction  

% Written by A. Karshenas -- Jan 31, 2024
%----------------------------------------------------


% Adjust data to span data range.
X = imadjust(X);

% Auto clustering
s = rng;
rng('default');
L = imsegkmeans(single(X),2,'NumAttempts',2);
rng(s);
BW = L == 2;

% Dilate mask with default
radius = 3;
decomposition = 0;
se = strel('disk', radius, decomposition);
BW = imdilate(BW, se);

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

