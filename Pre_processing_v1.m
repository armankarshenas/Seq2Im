Path_to_overhead_directory = "/media/zebrafish/Seagate1/seagate1";
img_patch_size = 10000;

cd(Path_to_overhead_directory)
logs = "";
DIRS = dir(pwd);
f = waitbar(0,'1');
for d=3:length(DIRS)
    if DIRS(d).isdir == 1
        cd(DIRS(d).folder)
        fprintf("Processing ... %s",DIRS(d).name)
        imgs = dir(fullfile(pwd,"*.tif"));
        for img=1:length(imgs)
            str_to_print = "Processing ... "+ imgs(img).name;
            waitbar(img/length(imgs),f,str_to_print);
            I = imread(imgs(img).name);
            mkdir(imgs(img).name)
            n = floor(size(I,1)/img_patch_size);
            m = floor(size(I,2)/img_patch_size);
            for i=1:n
                for j=1:m
                    I_local = I((i-1)*img_patch_size+1:i*img_patch_size,(j-1)*img_patch_size+1:j*img_patch_size);
                    coordinates = segmentImage(I_local);
                    write_images(coordinates,imgs(img).name,Path_to_overhead_directory);
                end
                I_local = I((i-1)*img_patch_size:i*img_patch_size,j*img_patch_size+1:end);
                coordinates = segmentImage(I_local);
                write_images(coordinates,imgs(img).name,Path_to_overhead_directory)
            end

        end
    end
end

