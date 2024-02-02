function coordinates = segmentImage_v2(X)

% segmentImage Segment image and return a structure of embryos with bounding 
% boxes around embryos that meet an area restriction  

% Written by A. Karshenas -- Feb 1, 2024
%----------------------------------------------------


% Adjust data to span data range.
X = imadjust(X);

% Auto clustering
s = rng;
rng('default');
L = imsegkmeans(single(X),2,'NumAttempts',2);
rng(s);
BW = L == 2;

% Watersheding the image and closing the holes 
A = watershed(BW);
A = imfill(A,'holes');

% Eroding the image with a disk 
radius = 3;
decomposition = 0;
se = strel('disk', radius, decomposition);
A = imerode(A,se);

% Creating a mask for X
A(A==1) = 0;
A(A~=0) = 1;
X = X.*uint8(A);

% Getting the clusters again 
s = rng;
rng('default');
L = imsegkmeans(single(X),2,'NumAttempts',2);
rng(s);
BW = L == 2;

maskedImage = X;
maskedImage(~BW) = 0;

% Create the structure for the embryos 
labeledImage = bwlabel(BW);
embryos = regionprops(labeledImage, 'BoundingBox');

coordinates = [];
for i = 1:numel(embryos)
    area = embryos(i).BoundingBox(3)*embryos(i).BoundingBox(4);
    if area > 150000 && area <400000
        coordinates = [coordinates;embryos(i).BoundingBox];
    end
end
end