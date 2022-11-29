function [mask_info, mask, nvox] = set_mask(path_mask)
    try
        mask_info = niftiinfo(path_mask);
        mask = uint8(niftiread(mask_info));
    catch ME
        throw(ME);
    end
end