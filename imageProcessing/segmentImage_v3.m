function [embryos,coordinates] = segmentImage_v3(X)

% segmentImage Segment image and return a structure of embryos with bounding 
% boxes around embryos that meet an area restriction  

% Written by A. Karshenas -- Feb 1, 2024
%----------------------------------------------------
s = rng;
rng('default');
L = imsegkmeans(single(X),2,'NumAttempts',2);
rng(s);
BW = L == 2;

A = watershed(BW);

radius = 3;
decomposition = 0;
se = strel('disk', radius, decomposition);
A_fill = imfill(A,'holes');
A_close = imclose(A_fill,se);
A_fill = imfill(A_close,'holes');
A_close = imclose(A_fill,se);
A_ero = imerode(A_close,se);

A_ero(A_ero==1) = 0;
A_ero(A_ero~=0) = 1;
mask = A_ero;

mask = uint8(mask);
X = mask.*X;

s = rng;
rng('default');
L = imsegkmeans(single(X),2,'NumAttempts',2);
rng(s);
BW = L == 2;

% Dilate mask with default
BW = imdilate(BW, se);

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
