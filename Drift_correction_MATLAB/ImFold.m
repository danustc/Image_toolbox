% Large, sparse image pre-processing and fold 

function Imf=ImFold(Im0, nfold)
% Im0: the original image 
% nfold: the iteration of fold, by default nfold =2 
if(nargin<2)
    nfold=2;
end

dim=size(Im0); 

rdim = 2^(nfold-1); % the real dimension that the folding needed 

if(mod(dim(1),rdim) ==0)
    exrow = dim(1);
else
    rowf=ceil(dim(1)/rdim);
    exrow=rowf*rdim; % add a number of empty rows 
end

if(mod(dim(2),rdim)==0)
    excol=dim(2);
else
    colf=ceil(dim(2)/rdim);
    excol = colf*rdim; % add a number of empty cols
end % endif 

Imf=zeros(exrow, excol);
Imf(1:dim(1),1:dim(2)) = Im0;

ifold = nfold;
while(ifold>1)
    Imf= Foldmat(Imf);
    ifold = ifold -1;
end


end


function Imf=Foldmat(Im0)
    % suppose each dimension's size is already an even number. 
    [nr,nc]=size(Im0);
    hnr = nr/2;
    hnc = nc/2;
    
    blkr= 1:hnr;
    blkc= 1:hnc;
    
    
    Imf = Im0(blkr, blkc)+Im0(blkr+hnr, blkc)+ Im0(blkr,blkc+hnc) ...
        + Im0(blkr+hnr, blkc+hnc);
    
end