% Pre_processing_v1 Segment image and writes images of single embryos with 
% bounding boxes around embryos that meet an area restriction  

% Written by A. Karshenas -- Jan 31, 2024
%----------------------------------------------------

%% Specifications 
Path_to_overhead_directory = "/media/zebrafish/Seagate1/seagate1";
Path_to_code = "";
addpath(genpath(Path_to_code));
img_patch_size = 10000;

%% Main code
cd(Path_to_overhead_directory)
logs = "";
DIRS = dir(pwd);
f = waitbar(0,'1');
for d=3:length(DIRS)
    if DIRS(d).isdir == 1
        cd(DIRS(d).folder)
        fprintf("Processing ... %s",DIRS(d).name)
        imgs = dir(fullfile(pwd,"*.tiff"));
        for img=1:length(imgs)
            str_to_print = "Processing ... "+ imgs(img).name;
            waitbar(img/length(imgs),str_to_print);
            I = imread(imgs(img).name);
            folder_name = split(imgs(img).name,".");
            folder_name = folder_name{1};
            mkdir(folder_name)
            n = floor(size(I,1)/img_patch_size);
            m = floor(size(I,2)/img_patch_size);
            % For the tiles that are patch_size x patch size 
            for i=1:n
                for j=1:m
                    I_local = I((i-1)*img_patch_size+1:i*img_patch_size,(j-1)*img_patch_size+1:j*img_patch_size);
                    coordinates = segmentImage(I_local);
                    writeImages(I_local,coordinates,folder_name,Path_to_overhead_directory,img_patch_size);
                end
                I_local = I((i-1)*img_patch_size+1:i*img_patch_size,j*img_patch_size+1:end);
                coordinates = segmentImage(I_local);
                writeImages(I_local,coordinates,folder_name,Path_to_overhead_directory,img_patch_size)
            end
            % For the final row of tiles that have a height less than
            % patch_size
            for j=1:m
                I_local = I(n*img_patch_size+1:end,(j-1)*img_patch_size+1:j*img_patch_size);
                coordinates =  segmentImage(I_local);
                writeImages(I_local,coordinates,folder_name,Path_to_overhead_directory,img_patch_size);
            end
            % For the final row-final column tile that is less than patch
            % size x patch_size
            I_local = I(n*img_patch_size+1:end,m*img_patch_size+1:end);
            coordinates =  segmentImage(I_local);
            writeImages(I_local,coordinates,folder_name,Path_to_overhead_directory,img_patch_size);
            
        end
    end
end

