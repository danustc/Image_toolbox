% path = '\\tsclient\X\\2016-05-18\B2_TS1\';
path = '\\tsclient\X\Zebrafish_ispim\2016-03-21\A2_3_TS2\';
% subpath = 'A1_2_TS2';
% pathall=[path subpath];

for i=0:399
    fmark = ['A2_3_TS2_TP_', num2str(i)];
    fsave = [path strcat(fmark,'_aligned')];
%     Im_align=Stack_volumealign(path, fmark, 31, fsave, 2);
    Im_align=Stack_volumealign(path, fmark, 27, fsave,2);
end