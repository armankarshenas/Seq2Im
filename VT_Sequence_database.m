%% Sequence parser file for VT dataset 
% This script parse the dm3 whole genome seqeunce of D. melanogaster 
% and extracts the sequence for each of the VT identifier 
% in the fly enhancer database out of Stark group in Austria 
clc 
clear 
% Specifying the path to files 
path_to_sequence = "~/Arman/BerkeleyPhD/Yr3/Seq2IM/VT-data/dm3.fa";
path_to_vt_coordinates = "~/Arman/BerkeleyPhD/Yr3/Seq2IM/VT-data/vt-genomic-coordinates.csv";
path_to_raw_metadata = "~/Arman/BerkeleyPhD/Yr3/Seq2IM/VT-data/raw_images_metadata.xlsx";
path_to_save = "~/Arman/BerkeleyPhD/Yr3/Seq2IM/VT-data";
% Reading the files 
seq = fastaread(path_to_sequence);
TB = readtable(path_to_vt_coordinates);
TB_to_write = readtable(path_to_raw_metadata);

tb_seq = struct2table(seq);
VTs_not = [];
for i=1:height(TB) 
    identifier_vt = string(TB.VTID(i));
    identifier = split(identifier_vt,"VT");
    identifier = str2double(identifier{2});
    chrom = string(TB.Chrosome(i));
    start = TB.Start(i);
    finish = TB.End(i);
    sequence = upper(string(tb_seq{tb_seq.Header == chrom,2}));
    seq_to_write = char(sequence);
    seq_to_write = seq_to_write(start+1:finish);
    idx = TB_to_write{:,2} == identifier;
    if any(idx)
        TB_to_write{idx,'Sequence'} = {seq_to_write};
    else 
        fprintf("The VT %s is not in image metadata \n",identifier_vt)
    end

end
cd(path_to_save)
writetable(TB_to_write,"Processed-meta-data.csv");

