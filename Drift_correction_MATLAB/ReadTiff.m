function im_stack=ReadTiff(fpath,fmark,L)
    % edited by Dan on 02-22-16
    % path: red a sequence of single files or a stack
    % single files test
    addpath(fpath);
    dirname=[fpath, '*', fmark, '.tif'];
    imFiles=dir(dirname); % is it a stack?
    
    if(isempty(imFiles))
        disp('No file found!');
        return;
        
    end
    
    Nim=length(imFiles);
    
    
    if(Nim>1) % if it is not a stack but a series of single-frame images
        im_ref=imread(imFiles(1).name); % have a reference image
        sref=size(im_ref);
        Ntrue=min(Nim,L);
        
        
        im_stack=zeros(sref(1),sref(2),Ntrue);
        
        
        im_stack(:,:,1)=im_ref;
        for il=2:Ntrue
            im_stack(:,:,il)=imread(imFiles(il).name);
        end
        
    else % if it is a stack, then read the first L frames
        
        im_stack=TiffStack(fpath,imFiles.name, L);
        % read a stack
    end
    
end

% InfoImage=imfinfo(imFiles.name);
% mImage=InfoImage(1).Width;
% nImage=InfoImage(1).Height;
% NumIm=length(InfoImage);
%
% if(NumIm > L)
%     im_stack=zeros(nImage,mImage, NumIm, 'uint16');
% end
%
% TifLink=Tiff(imFiles.name,'r');
%
% for il=1:L
%     TifLink.setDirectory(il);
%     im_stack(:,:,il)=TifLink.read();
% end
%
% TifLink.close();
% end
