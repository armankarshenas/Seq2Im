function [] = write_images(I,coordinates,name,path)
for i=1:size(coordinates,1)
    sp(1) = max(floor(coordinates(i,1)), 1); %xmin
    sp(2) = max(floor(coordinates(i,2)), 1);%ymin
    sp(3)= min(ceil(coordinates(i,1) + coordinates(i,3))); %xmax
    sp(4)=min(ceil(coordinates(i,2) +coordinates(i,4))); %ymax
    embryo_im = I(sp(2):sp(4), sp(1): sp(3));
    image(embryo_im)
    % Write image to graphics file.
    cd(path + "/"+name)
    files = dir(fullfile(pwd,"*.tif"));
    if isempty(files)
        counter = 1;
    else 
        last_im_in_dir = files(length(files)).name;
        last_im_in_dir = split(last_im_in_dir,"_");
        counter = last_im_in_dir{2};
        counter = split(counter,".");
        counter = str2double(counter{1});
    end
    f_name = name+"_"+string(counter+i)+".tif";
    imwrite(embryo_im,f_name)
end
end