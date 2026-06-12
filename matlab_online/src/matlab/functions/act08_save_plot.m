function act08_save_plot(figHandle, outFile)
%ACT08_SAVE_PLOT Saves a MATLAB figure with a fallback for older releases.

folderPath = fileparts(outFile);
if ~isfolder(folderPath)
    mkdir(folderPath);
end

try
    exportgraphics(figHandle, outFile, 'Resolution', 180);
catch
    saveas(figHandle, outFile);
end
end
