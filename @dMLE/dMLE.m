classdef dMLE
    %dMLE provides tools for Maximum Likelihood Estimation of
    %diffusion-weighted MRI models. Currently supported models are
    %diffusion tensor and kurtosis tensor.
    %
    % Viljami Sairanen
    
    % Example
%     dMLE('G:\dMRI_datasets\HCP\103818\dwi_preproc_B1corr.nii.gz', ...
%     'DKI', ...
%     'G:\dMRI_datasets\HCP\103818\nodif_brain_mask.nii.gz', ...
%     'G:\dMRI_datasets\HCP\103818\bval', ...
%     'G:\dMRI_datasets\HCP\103818\bvec')
%     
    properties (Access = public)
        path_dwi;
        model;
        path_bval;
        path_bvec;
        path_mask;
        X single; % design matrix        
        bval;
        bvec;
        N uint32;
        dwi single;
        dwi_info;
        mask uint8;
        mask_info;
        % MLE options
        tolerance_sigmasq       single= 1e-4;
        tolerance_theta         single= 1e-4;
        tolerance_S0            single= 1e-4;
        tolerance_loglikelihood single= 1e-5;
        % tolerance_rice = 1e-27
        iter_limit_sigmasq      uint32= 100;
        iter_limit_S0           uint32= 100;
        iter_limit_voxel        single= 10;
        iter_limit_theta        single= 10;
        iter_limit              uint32= 50; % This can't be large due regulator lambda (1.0e-5*5^(iter))
        lambda0                 single= 1e-5;
        rescaling               single= 5.0;
        % MLE params
        sumYSQ single
        iTheta single
        iSigmaSQ single
        mlTheta single
        mlSigmaSQ single
    end
    
       
    methods (Static, Access = public)
        [bval, bvec, N] = set_bval_bvec(path_bval, path_bvec);
        X = set_design_matrix(bval, bvec, model);
        [mask_info, mask, nvox] = set_mask(path_mask);
        [dwi_info, dwi] = set_dwi(path_dwi);
    end
    
    methods (Access = public)
        
        function obj = dMLE(path_dwi, model, path_mask, path_bval, path_bvec)

            if nargin == 2
                obj.path_dwi = path_dwi;
                obj.model = model;
                obj.path_bval = [path_dwi(1:end-7), '.bval'];
                obj.path_bvec = [path_dwi(1:end-7), '.bvec'];
            elseif nargin == 3
                obj.path_dwi = path_dwi;
                obj.model = model;
                obj.path_bval = [path_dwi(1:end-7), '.bval'];
                obj.path_bvec = [path_dwi(1:end-7), '.bvec'];
                obj.path_mask = path_mask;
            elseif nargin == 5
                obj.path_dwi = path_dwi;
                obj.model = model;
                obj.path_mask = path_mask;
                obj.path_bval = path_bval;
                obj.path_bvec = path_bvec; 
            else
                disp('Error in dMLE initialization');
                quit;
            end
            
            [obj.bval, obj.bvec, obj.N] = set_bval_bvec(obj.path_bval, obj.path_bvec);
            obj.X = set_design_matrix(obj.bval, obj.bvec, obj.model);
            [obj.mask_info, obj.mask] = set_mask(obj.path_mask);
            [obj.dwi_info, obj.dwi] = set_dwi(obj.path_dwi);
            
            if nargout == 0
                clear obj;
            end
        end
        
        function dMLE_save(obj, output_prefix)            
            % transform vectorized data to 3D
            mask = reshape(obj.mask, obj.mask_info.ImageSize) > 0;
            inds = find(mask);
            sigmas = zeros(obj.mask_info.ImageSize, 'single');
            tmp = zeros(obj.mask_info.ImageSize, 'single');
            thetas = zeros([obj.mask_info.ImageSize, size(obj.X, 2)], 'single');
            sigmas(inds) = obj.mlSigmaSQ;
            for i = 1:size(obj.X, 2)
                tmp(inds) = obj.mlTheta(i,:);
                thetas(:,:,:,i) = tmp;
            end
            sigma_info = obj.mask_info;
            theta_info = obj.dwi_info;
            theta_info.ImageSize(4) = size(obj.X, 2);
            niftiwrite(thetas, [output_prefix, '_params'], theta_info, 'Compressed', true);
            niftiwrite(sigmas, [output_prefix, '_sigmas'], sigma_info, 'Compressed', true);
        end
                
        function obj = dMLE_init(obj)
            if size(obj.mask,2) > 1
                % Mask dwi to remove non-brain voxels
                obj.dwi = obj.dwi(repmat(obj.mask, 1, obj.dwi_info.ImageSize(4)) > 0);
                % Vectorize mask and dwi
                obj.mask = obj.mask(:);
                n = sum(obj.mask);
                obj.dwi = reshape(obj.dwi, n, ...
                                        obj.dwi_info.ImageSize(4))';
            end
            % Calculate initial guess for MLE
            obj.sumYSQ = sum( obj.dwi.^2, 1);
            obj.iTheta = obj.X \ log(obj.dwi + eps);
            obj.iSigmaSQ = sum(( obj.dwi - exp( obj.X * obj.iTheta )).^2); 
        end
        
        function obj = dMLE_fit(obj)
            
            nVoxels = size(obj.dwi,2);
            nDWIs = size(obj.dwi,1);
            G = gpuDevice(1);
            reset(G); % Just in case...
            % if G.KernelExecutionTimeout
            %     disp('Kernel Execution Timeout detected, please ensure that your OS allows long GPU processess!');
            % end
            cudaFilename = 'RicianMLE_single.cu';
            ptxFilename = 'RicianMLE_single.ptx';
            kernel = parallel.gpu.CUDAKernel( ptxFilename, cudaFilename );

            obj.mlTheta = single(obj.iTheta);
            obj.mlSigmaSQ = single(obj.iSigmaSQ);
            nParams = size(obj.X, 2);
            nDeltaParams = nParams - 1;
            nCalc = ceil(nVoxels/10); % adjust 10 to higher number if memory runs out

            fprintf(1, '\nDone:   %d%s\nTime remaining: ???:??\nEstimated time: ???:??', 0, '%');
            tic
            for i = 1:ceil(nVoxels/nCalc)   
                % Split data into blocks so GPU doesn't run out of memory
                blockInds = ((i-1)*nCalc+1:(i*nCalc));    
                blockInds(blockInds > nVoxels) = [];
                blockTheta = single(obj.iTheta( :, blockInds ));
                blockSigmaSQ = single(obj.iSigmaSQ( blockInds ));
                blockY = obj.dwi(:, blockInds);
                blockSumYSQ = obj.sumYSQ( blockInds );
                blockVoxs = length(blockInds);

                % Initialize arrays needed in GPU memory
                fisherInformation = zeros(nDeltaParams^2, blockVoxs);
                fisherInformation_sym = zeros(nDeltaParams^2, blockVoxs);
                score = zeros(nDeltaParams,blockVoxs);
                DeltaTheta = zeros(nDeltaParams, blockVoxs);
                new_theta = zeros(nDeltaParams, blockVoxs);
                vv = zeros(nDeltaParams, blockVoxs);
                indx = zeros(nDeltaParams, blockVoxs);
                theta_old = zeros(nParams,blockVoxs);
                expZTheta = zeros(nDWIs, blockVoxs);
                ZTheta = zeros(nDWIs, blockVoxs);
                twotau = zeros(nDWIs,blockVoxs);
                expo = zeros(nDWIs, blockVoxs);
                EN = zeros(nDWIs,blockVoxs);
                b = zeros(nDWIs, blockVoxs);
                a = zeros(blockVoxs,1);
                c = zeros(blockVoxs,1);  
                theta1_old = zeros(blockVoxs,1);
                SigmaSQ0 = zeros(blockVoxs,1);
                SigmaSQ_old = zeros(blockVoxs,1);
                tmpvar = zeros(blockVoxs,1);
                scaling = zeros(blockVoxs,1);
                expScaling = zeros(blockVoxs,1);
                loglikelihood = zeros(blockVoxs,1);
                loglikelihood_old = zeros(blockVoxs,1);
                regulatorLambda = zeros(blockVoxs,1);
                DeltaThetaScore = zeros(blockVoxs,1);
                norm1 = zeros(blockVoxs,1);
                norm2 = zeros(blockVoxs,1);
                obj.tolerance_sigmasq;
                obj.tolerance_S0;
                obj.tolerance_theta;
                obj.tolerance_loglikelihood;
                nIterSigmaSQ = zeros(blockVoxs,1);
                nIterVoxel = zeros(blockVoxs,1);
                nIterS0 = zeros(blockVoxs,1);
                nIterTheta = zeros(blockVoxs,1);
                nIterLoglikelihood = zeros(blockVoxs,1);
                continueSigmaSQIteration = zeros(blockVoxs,1);
                continueVoxelIteration = zeros(blockVoxs,1);
                continueS0Iteration = zeros(blockVoxs,1);
                continueThetaIteration = zeros(blockVoxs,1);
                continueLoglikelihoodIteration = zeros(blockVoxs,1);
                anyEN = zeros(blockVoxs,1);
                obj.lambda0;
                obj.rescaling;
%                 nDWIs;
%                 nParams;
%                 nDeltaParams;
%                 blockVoxs;

                % Initialise CUDA kernel
            %     reset(G); % Just in case...
            %     kernel = parallel.gpu.CUDAKernel( ptxFilename, cudaFilename ); 
            %     kernel.ThreadBlockSize = [kernel.MaxThreadsPerBlock, 1, 1];
                kernel.ThreadBlockSize = [kernel.MaxThreadsPerBlock, 1];
            %     kernel.ThreadBlockSize = [64, 1];
                kernel.GridSize = [ceil(blockVoxs/kernel.ThreadBlockSize(1)), 1];

                [blockTheta, blockSigmaSQ] = feval( kernel,...
                single(blockTheta), ...
                single(blockSigmaSQ), ...
                single(obj.X), ...
                single(fisherInformation), ...
                single(fisherInformation_sym), ...
                single(score), ...
                single(DeltaTheta), ...
                single(new_theta), ...
                single(vv), ...
                single(indx), ...
                single(theta_old), ...
                single(blockY), ...
                single(expZTheta), ...
                single(ZTheta), ...
                single(twotau), ...
                single(expo), ...
                single(EN), ...
                single(b), ...
                single(a), ...
                single(c), ...  
                single(blockSumYSQ), ...
                single(theta1_old), ...
                single(SigmaSQ0), ...
                single(SigmaSQ_old), ...
                single(tmpvar) ,...
                single(scaling), ...
                single(expScaling), ...
                single(loglikelihood), ...
                single(loglikelihood_old), ...
                single(regulatorLambda), ...
                single(DeltaThetaScore), ...
                single(norm1), ...
                single(norm2), ...
                single(nIterSigmaSQ), ...
                single(nIterVoxel), ...
                single(nIterS0), ...
                single(nIterTheta), ...
                single(nIterLoglikelihood), ... 
                single(continueSigmaSQIteration), ...
                single(continueVoxelIteration), ...
                single(continueS0Iteration), ...
                single(continueThetaIteration), ...
                single(continueLoglikelihoodIteration), ...
                single(anyEN), ...                                    
                single(obj.tolerance_sigmasq), ...
                single(obj.tolerance_S0), ...
                single(obj.tolerance_theta), ...
                single(obj.tolerance_loglikelihood), ...
                single(obj.iter_limit_sigmasq) ,...
                single(obj.iter_limit_voxel) ,...
                single(obj.iter_limit_S0) ,...
                single(obj.iter_limit_theta) ,...
                single(obj.iter_limit) ,...
                single(obj.lambda0), ...
                single(obj.rescaling), ...
                single(nDWIs), ...
                single(nParams), ...
                single(nDeltaParams), ...
                single(blockVoxs));

                obj.mlTheta(:, blockInds) = gather(blockTheta);
                obj.mlSigmaSQ(blockInds) = gather(blockSigmaSQ);   

                timesofar = toc;     
                totalTimeSeconds = timesofar / blockInds(end) * nVoxels;
                TimeEstimateMinutes = floor(totalTimeSeconds / 60);
                TimeEstimateSeconds = round(totalTimeSeconds - TimeEstimateMinutes*60);
                TimeRemainingMinutes = floor((totalTimeSeconds-timesofar)/60);
                TimeRemainingSeconds = round(totalTimeSeconds-timesofar-TimeRemainingMinutes*60);

                if TimeEstimateSeconds < 10
                    strSest = ['0', num2str(TimeEstimateSeconds)];
                else
                    strSest = num2str(TimeEstimateSeconds);
                end
                if TimeRemainingSeconds < 10
                    strSrem = ['0', num2str(TimeRemainingSeconds)];
                else
                    strSrem = num2str(TimeRemainingSeconds);
                end
                if TimeEstimateMinutes < 10
                    strMest = ['  ', num2str(TimeEstimateMinutes)];
                elseif TimeEstimateMinutes < 100
                    strMest = [' ', num2str(TimeEstimateMinutes)];
                else
                    strMest = num2str(TimeEstimateMinutes);
                end
                if TimeRemainingMinutes < 10
                    strMrem = ['  ', num2str(TimeRemainingMinutes)];
                elseif TimeRemainingMinutes < 100
                    strMrem = [' ', num2str(TimeRemainingMinutes)];
                else
                    strMrem = num2str(TimeRemainingMinutes);
                end    
                donePercentage = round(i/ceil(nVoxels/blockVoxs)*100);
                if donePercentage < 10
                    strD = ['  ', num2str(donePercentage)];
                elseif donePercentage < 100
                    strD = [' ', num2str(donePercentage)];
                else
                    strD = num2str(donePercentage);
                end
                fprintf(1, [repmat('\b', [1,56]), 'Done: %s%s\nTime remaining: %s:%s\nEstimated time: %s:%s'], strD, '%', strMrem, strSrem, strMest, strSest);

            end
            fprintf(1, '\n');
    
        end
            
    end
       
       
    
end

