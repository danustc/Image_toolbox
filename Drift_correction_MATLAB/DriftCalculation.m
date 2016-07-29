function [drift, Ccorr_map]=DriftCalculation(Im_ref, Im_shif, mfit)
    % Drift correction between two images
    % Im_ref: the reference image
    % Im_shif: the shifted image,translation only
    % drift: calculated drift, unit in pixels
    % mfit: to what extent the maximum of the correlation is fitted
    % by default the two images are in the same size.
    % This one only calculates the drift, no corrections.
    
    if(nargin<3)
        mfit=0;
    end
    
    idim = size(Im_ref);
    
    FT_ref = fft2(Im_ref);
    FT_shif = fft2(Im_shif);
    
    Cxy=ifft2(conj(FT_ref).*FT_shif);
    
    [~, nind] = max(Cxy(:));
    
    [Shy, Shx] = ind2sub(idim, nind); 
  
    function indo=res_ind(indi, hdim)
        % hdim: the dimension of the array. If the indi is larger than half of hdim, then indi is replaced by indi-hdim.
        if(indi > hdim/2)
            indo = indi - hdim;
        else
            indo = indi;
        end
        
        
    end
    
    dry=res_ind(Shy,idim(1));
    drx=res_ind(Shx,idim(2));
    
    
    % drift y, drift x for row, column shifts
    % Warning: because matlab arrays start indexing from 1 instead of 0,
    % the real shift center should be (drx, dry) -1.
    
    
    if(mfit==0)
        drift=[dry drx]-1;
        Ccorr_map=Cxy;
        % from circshifting by drift, then it can be restored by circshifting for -drift.
    else % mfit doesn't equal to 0
        % So here we need to discuss if the shifted center [dry drx] is  on
        % the edge, i.e., if the shift is slight or significant. for slight
        % shifts, Cxy must be fftshifted before Gaussian fitting for the center search. 
        
        ncomp=round(idim/4); % the comparison factor 
        if(abs(drx)<ncomp(2))
            % shift in x 
            Cxy=circshift(Cxy, [0 ncomp(2)*2]); % it's like fftshift in one direction only
            Cnx=drx+ncomp(2)*2; 
        else 
            Cnx=mod(drx, idim(2));
            % do not circshift in x 
        end  
        
        
        if(abs(dry)<ncomp(1))
            % circshift in y
            Cxy=circshift(Cxy, [ncomp(1)*2 0]);
            Cny=dry+ncomp(1)*2;
        else
            Cny=mod(dry, idim(1));
            % do not circshift in y
        end
        
        
        
        
        xrange = (Cnx-mfit):(Cnx+mfit);
        yrange = (Cny-mfit):(Cny+mfit);
        
       
        
        Ccorr_map=Cxy;
        corrprofile = Cxy (yrange,xrange); % Take out the brightest block of the correlation function
        gfit = gauss2dfit(corrprofile);
        rshift = gfit.e;
        cshift = gfit.c;
        drift = [dry drx] + round([rshift cshift])-1;
        
    end
end

