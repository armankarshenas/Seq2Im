function [] = writeImages(I,coordinates,name,path,img_patch_size)

% writeImages uses the bouding boxes generated by segmentImage to crop 
% single embryos out of the microscopy image and writes them as separate files 
% in the specified directory 

% Written by A. Karshenas -- Jan 31, 2024
%----------------------------------------------------


cd(path + "/"+name)
files = dir(fullfile(pwd,"*.tif"));
if isempty(files)
    counter = 0;
else
    last_im_in_dir = files(length(files)).name;
    last_im_in_dir = split(last_im_in_dir,"_");
    counter = last_im_in_dir{2};
    counter = split(counter,".");
    counter = str2double(counter{1});
end

if size(coordinates,1)>1
    for i=1:size(coordinates,1)
        sp(1) = max(floor(coordinates(i,1)), 1); %xmin
        sp(2) = max(floor(coordinates(i,2)), 1);%ymin
        sp(3)= min(ceil(coordinates(i,1) + coordinates(i,3)),img_patch_size); %xmax
        sp(4)=min(ceil(coordinates(i,2) +coordinates(i,4)),img_patch_size); %ymax
        embryo_im = I(sp(2):sp(4), sp(1): sp(3));
        image(embryo_im)
        % Write image to graphics file.

        f_name = name+"_"+string(counter+i)+".tif";
        imwrite(embryo_im,f_name)
    end
end
end