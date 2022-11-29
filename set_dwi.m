function [dwi_info, dwi] = set_dwi(path_dwi)
    try
        dwi_info = niftiinfo(path_dwi);
        dwi = single(niftiread(dwi_info));
    catch ME
        throw(ME);
    end
end