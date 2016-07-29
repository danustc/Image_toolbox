function Im_align=Stack_driftalign(path, fmark, nread, fsave)
    % path: file path where the images to be loaded are
    % fmark: the characteristic string in the file name
    % fsave: the filename to be saved
    % nread: number of the files to be read
    %
    % ---------Part 1: load the tiff image-----------------
    
    
    addpath(path);
    Im_stack=ReadTiff(path,fmark, nread);
    Idim  = size(Im_stack);
    Im_align = zeros(Idim);
    
    Im_ref = Im_stack(:,:,1);
    Im_align(:,:,1)=Im_ref;
    
    for k = 2: nread
        Im_k=Im_stack(:,:,k);
        drift = DriftCalculation(Im_ref, Im_k,0);
%         imagesc(real(ICorr)); % I just want to check the corr-function.
        disp(['--------image ' num2str(k) '--------:' num2str(drift)]);
        Im_align(:,:,k) = circshift(Im_k, -drift);
    end
    
    Im_align=uint16(Im_align);
    if(nargin==4)
        outputFileName = [fsave '.tif'];
%         twrite=Tiff(outputFileName, 'a');
%         
%         tagstruct.ImageLength     = Idim(1);
%         tagstruct.ImageWidth      = Idim(2);
%         tagstruct.Photometric     = Tiff.Photometric.MinIsBlack;
%         tagstruct.BitsPerSample   = 16;
%         tagstruct.SamplesPerPixel = 1;
%         tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
%         tagstruct.Software        = 'MATLAB';
%         twrite.setTag(tagstruct);
        
        for k=1:nread
%             twrite.write(Im_align(:,:,k));
            imwrite(Im_align(:,:,k), outputFileName, 'WriteMode', 'append');
        end
%         twrite.close();
    end
    
   
    
end