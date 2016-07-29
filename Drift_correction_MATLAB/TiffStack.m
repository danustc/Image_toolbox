function FinalImage=TiffStack(path, FileTif, L)
    
    % path = 'D:\Dan\Data\Processing_0211\';
    addpath(path);
    % tic;
%     FileTif='Feb10_N1TS2_slice17.tif';
    InfoImage=imfinfo(FileTif);
    mImage=InfoImage(1).Width;
    nImage=InfoImage(1).Height;
    NumberImages=min(L,length(InfoImage));
    FinalImage=zeros(nImage,mImage,NumberImages,'uint16');
    FileID = tifflib('open',[path, FileTif],'r');
    rps = tifflib('getField',FileID,Tiff.TagID.RowsPerStrip);
    
    
    
    for i=1:NumberImages
        tifflib('setDirectory',FileID,i);
        % Go through each strip of data.
        rps = min(rps,nImage);
        for r = 1:rps:nImage
            row_inds = r:min(nImage,r+rps-1);
            stripNum = tifflib('computeStrip',FileID,r);
            FinalImage(row_inds,:,i) = tifflib('readEncodedStrip',FileID,stripNum);
        end
    end
    tifflib('close',FileID);
end
