function [bval, bvec, N] = set_bval_bvec(path_bval, path_bvec)
    try
        bval = load(path_bval);
        bvec = load(path_bvec);
        if size(bvec,2) ~= 3
            bvec = bvec';
        end
        N = length(bval);
    catch ME
        throw(ME);
    end   
end