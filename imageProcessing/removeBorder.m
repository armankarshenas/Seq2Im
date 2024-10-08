function I_sub = removeBorder(I)

% removeBorder uses the histogram count of intensity to determine if
% borders exist. The function then truncates image I so it can be used for
% downstream processing 

% Written by A. Karshenas -- Jan 31, 2024
%----------------------------------------------------

% horizontal profiles
left_border = 0;
right_border = 0;
top_border = 0;
bottom_border = 0;

idx_x = floor(linspace(2000,size(I,1)-2000,20));
for i=1:20
    horizontal_profile = I(idx_x(i),:);
    h_changes = ischange(double(horizontal_profile));
    if find(h_changes,1) > left_border && left_border < 2000
        left_border = find(h_changes,1);
    end
    if find(flip(h_changes),1) > right_border && right_border < 2000
    right_border = find(flip(h_changes),1);
    end
end

idx_y = floor(linspace(2000,size(I,2)-2000,20));
for i=1:20
vertical_profile = I(:,idx_y(i));
v_changes = ischange(double(vertical_profile));
if find(v_changes,1) > top_border && top_border <2000
top_border = find(v_changes,1);
end
if find(flip(v_changes),1) > bottom_border && bottom_border<2000
bottom_border = find(flip(v_changes),1);
end
end

I_sub = I(top_border:size(I,1)-bottom_border,left_border:size(I,2)-right_border);
end