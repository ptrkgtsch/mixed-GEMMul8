function plot_flops(saveflag,savedir)

% close all

if nargin == 0
    saveflag = false;
end

%% files
fileList_f = dir("oz2_results_f_time_*");
filename_f = strings(length(fileList_f),1);
for i=1:length(fileList_f)
    filename_f(i) = string(fileList_f(i).name);
end
fileList_d = dir("oz2_results_d_time_*");
filename_d = strings(length(fileList_d),1);
for i=1:length(fileList_d)
    filename_d(i) = string(fileList_d(i).name);
end
fileList_dfd = dir("oz2_results_dfd_time_*");
filename_dfd = strings(length(fileList_dfd),1);
for i=1:length(fileList_dfd)
    filename_dfd(i) = string(fileList_dfd(i).name);
end
fileList_dff = dir("oz2_results_dff_time_*");
filename_dff = strings(length(fileList_dff),1);
for i=1:length(fileList_dff)
    filename_dff(i) = string(fileList_dff(i).name);
end

%% float
for fn = 1:length(filename_f)
    filename = filename_f(fn);
    opts = detectImportOptions(filename);
    opts.SelectedVariableNames = 2;
    n = readmatrix(filename,opts);
    n_list = unique(n,'stable');
    opts.SelectedVariableNames = 5;
    func = string(readmatrix(filename,opts));
    func_list = unique(func,'stable');
    opts.SelectedVariableNames = 8;
    tflops = readmatrix(filename,opts);
    moduli_min = 2;
    moduli_max = 15;
    xx = moduli_min:moduli_max;
    XLIM = [2 12];
    
    fig = figure;
    fig.Position(3) = 1000;
    fig.Position(4) = 250;
    t = tiledlayout(1,length(n_list));
    for ni = 1:length(n_list)
        SGEMM    = tflops(n == n_list(ni) & func == "SGEMM");
        SGEMM_TF = tflops(n == n_list(ni) & func == "SGEMM-TF32");
        FP16TCEC = tflops(n == n_list(ni) & func == "FP16TCEC_SCALING");
        OS2_fast = tflops(n == n_list(ni) & contains(func,"OS2-fast"));
        OS2_accu = tflops(n == n_list(ni) & contains(func,"OS2-accu"));
        nexttile; hold on; grid on;
        plot(xx, SGEMM*ones(size(xx)), mark(1,1), 'DisplayName', "SGEMM", 'LineWidth',1);
        if ~isempty(SGEMM_TF)
            plot(xx, SGEMM_TF*ones(size(xx)), mark(5,1), 'DisplayName', "SGEMM-TF32", 'LineWidth',1);
        end
        if ~isempty(FP16TCEC)
            plot(xx, FP16TCEC*ones(size(xx)), mark(2,2), 'DisplayName', "cuMpSGEMM", 'LineWidth',1);
        end
        plot(xx, OS2_fast, mark(3,3), 'DisplayName', "OS II-fast", 'LineWidth',1);
        plot(xx, OS2_accu, mark(4,4), 'DisplayName', "OS II-accu", 'LineWidth',1);
        
        title("n=" + n_list(ni),'FontSize',14);
        ylim('padded');
        xlim(XLIM);
        xlabel("#moduli",'FontSize',14);
        set(gca,'FontSize',14);
    end
    lgd = legend;
    lgd.Layout.Tile = 'east'; %'south';
    lgd.NumColumns = 1; %length(func_list);
    lgd.FontSize = 14;
    pattern = "time_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    title(t, env, 'FontSize',14);
    ylabel(t, "TFLOPS",'FontSize',14);
    t.TileSpacing = "tight";
    t.Padding = "compact";
    
    if saveflag
        pattern = "(.*?)_2025";
        match = regexp(filename, pattern, 'tokens');
        figname = match{1}{1};
        savefig(fig,savedir+figname);
        exportgraphics(fig,savedir+figname + ".png",'Resolution',600);
    end
end

%% double
for fn = 1:length(filename_d)
    filename = filename_d(fn);
    opts = detectImportOptions(filename);
    opts.SelectedVariableNames = 2;
    n = readmatrix(filename,opts);
    n_list = unique(n,'stable');
    opts.SelectedVariableNames = 5;
    func = string(readmatrix(filename,opts));
    func_list = unique(func,'stable');
    opts.SelectedVariableNames = 8;
    tflops = readmatrix(filename,opts);
    moduli_min = 2;
    moduli_max = 20;
    xx = moduli_min:moduli_max;
    XLIM = [8 moduli_max];
    
    fig = figure;
    fig.Position(3) = 1000;
    fig.Position(4) = 250;
    t = tiledlayout(1,length(n_list));
    for ni = 1:length(n_list)
        DGEMM    = tflops(n == n_list(ni) & func == "DGEMM");
        ozIMMU_EF_8 = tflops(n == n_list(ni) & contains(func,"ozIMMU_EF-8"));
        ozIMMU_EF_9 = tflops(n == n_list(ni) & contains(func,"ozIMMU_EF-9"));
        OS2_fast = tflops(n == n_list(ni) & contains(func,"OS2-fast"));
        OS2_accu = tflops(n == n_list(ni) & contains(func,"OS2-accu"));
        nexttile; hold on; grid on;
        plot(xx, DGEMM*ones(size(xx)), mark(1,1), 'DisplayName', "DGEMM", 'LineWidth',1);
        if ~isempty(ozIMMU_EF_8)
            plot(xx, ozIMMU_EF_8*ones(size(xx)), mark(5,2), 'DisplayName', "ozIMMU\_EF-8", 'LineWidth',1);
        end
        if ~isempty(ozIMMU_EF_9)
            plot(xx, ozIMMU_EF_9*ones(size(xx)), mark(6,2), 'DisplayName', "ozIMMU\_EF-9", 'LineWidth',1);
        end
        plot(xx, OS2_fast, mark(3,3), 'DisplayName', "OS II-fast", 'LineWidth',1);
        plot(xx, OS2_accu, mark(4,4), 'DisplayName', "OS II-accu", 'LineWidth',1);
        
        title("n=" + n_list(ni),'FontSize',14);
        ylim('padded');
        xlim(XLIM);
        xticks(XLIM(1):2:XLIM(2));
        xlabel("#moduli",'FontSize',14);
        set(gca,'FontSize',14);
    end
    lgd = legend;
    lgd.Layout.Tile = 'east'; %'south';
    lgd.NumColumns = 1; %length(func_list);
    lgd.FontSize = 14;
    pattern = "time_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    title(t, env, 'FontSize',14);
    ylabel(t, "TFLOPS",'FontSize',14);
    t.TileSpacing = "tight";
    t.Padding = "compact";
    
    if saveflag
        pattern = "(.*?)_2025";
        match = regexp(filename, pattern, 'tokens');
        figname = match{1}{1};
        savefig(fig,savedir+figname);
        exportgraphics(fig,savedir+figname + ".png",'Resolution',600);
    end
end

%% f64-f32 -> f64
for fn = 1:length(filename_dfd)
    filename = filename_dfd(fn);
    opts = detectImportOptions(filename);
    opts.SelectedVariableNames = 2;
    n = readmatrix(filename,opts);
    n_list = unique(n,'stable');
    opts.SelectedVariableNames = 5;
    func = string(readmatrix(filename,opts));
    func_list = unique(func,'stable');
    opts.SelectedVariableNames = 8;
    tflops = readmatrix(filename,opts);
    moduli_min = 2;
    moduli_max = 20;
    xx = moduli_min:moduli_max;
    XLIM = [8 moduli_max];
    
    fig = figure;
    fig.Position(3) = 1000;
    fig.Position(4) = 250;
    t = tiledlayout(1,length(n_list));
    for ni = 1:length(n_list)
        DGEMM    = tflops(n == n_list(ni) & (func == "DGEMM" | func == "DSGEMM"));
        ozIMMU_EF_8 = tflops(n == n_list(ni) & contains(func,"ozIMMU_EF-8"));
        ozIMMU_EF_9 = tflops(n == n_list(ni) & contains(func,"ozIMMU_EF-9"));
        OS2_fast = tflops(n == n_list(ni) & contains(func,"OS2-fast"));
        OS2_accu = tflops(n == n_list(ni) & contains(func,"OS2-accu"));
        nexttile; hold on; grid on;
        plot(xx, DGEMM*ones(size(xx)), mark(1,1), 'DisplayName', "DGEMM", 'LineWidth',1);
        if ~isempty(ozIMMU_EF_8)
            plot(xx, ozIMMU_EF_8*ones(size(xx)), mark(5,2), 'DisplayName', "ozIMMU\_EF-8", 'LineWidth',1);
        end
        if ~isempty(ozIMMU_EF_9)
            plot(xx, ozIMMU_EF_9*ones(size(xx)), mark(6,2), 'DisplayName', "ozIMMU\_EF-9", 'LineWidth',1);
        end
        plot(xx, OS2_fast, mark(3,3), 'DisplayName', "OS II-fast", 'LineWidth',1);
        plot(xx, OS2_accu, mark(4,4), 'DisplayName', "OS II-accu", 'LineWidth',1);
        
        title("n=" + n_list(ni),'FontSize',14);
        ylim('padded');
        xlim(XLIM);
        xticks(XLIM(1):2:XLIM(2));
        xlabel("#moduli",'FontSize',14);
        set(gca,'FontSize',14);
    end
    lgd = legend;
    lgd.Layout.Tile = 'east'; %'south';
    lgd.NumColumns = 1; %length(func_list);
    lgd.FontSize = 14;
    pattern = "time_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    title(t, env, 'FontSize',14);
    ylabel(t, "TFLOPS",'FontSize',14);
    t.TileSpacing = "tight";
    t.Padding = "compact";
    
    if saveflag
        pattern = "(.*?)_2025";
        match = regexp(filename, pattern, 'tokens');
        figname = match{1}{1};
        savefig(fig,savedir+figname);
        exportgraphics(fig,savedir+figname + ".png",'Resolution',600);
    end
end

%% f64-f32 -> f32
for fn = 1:length(filename_dff)
    filename = filename_dff(fn);
    opts = detectImportOptions(filename);
    opts.SelectedVariableNames = 2;
    n = readmatrix(filename,opts);
    n_list = unique(n,'stable');
    opts.SelectedVariableNames = 5;
    func = string(readmatrix(filename,opts));
    func_list = unique(func,'stable');
    opts.SelectedVariableNames = 8;
    tflops = readmatrix(filename,opts);
    moduli_min = 2;
    moduli_max = 15;
    xx = moduli_min:moduli_max;
    XLIM = [2 12];
    
    fig = figure;
    fig.Position(3) = 1000;
    fig.Position(4) = 250;
    t = tiledlayout(1,length(n_list));
    for ni = 1:length(n_list)
        SGEMM    = tflops(n == n_list(ni) & func == "SGEMM");
        SGEMM_TF = tflops(n == n_list(ni) & func == "SGEMM-TF32");
        FP16TCEC = tflops(n == n_list(ni) & func == "FP16TCEC_SCALING");
        OS2_fast = tflops(n == n_list(ni) & contains(func,"OS2-fast"));
        OS2_accu = tflops(n == n_list(ni) & contains(func,"OS2-accu"));
        nexttile; hold on; grid on;
        plot(xx, SGEMM*ones(size(xx)), mark(1,1), 'DisplayName', "SGEMM", 'LineWidth',1);
        if ~isempty(SGEMM_TF)
            plot(xx, SGEMM_TF*ones(size(xx)), mark(5,1), 'DisplayName', "SGEMM-TF32", 'LineWidth',1);
        end
        if ~isempty(FP16TCEC)
            plot(xx, FP16TCEC*ones(size(xx)), mark(2,2), 'DisplayName', "cuMpSGEMM", 'LineWidth',1);
        end
        plot(xx, OS2_fast, mark(3,3), 'DisplayName', "OS II-fast", 'LineWidth',1);
        plot(xx, OS2_accu, mark(4,4), 'DisplayName', "OS II-accu", 'LineWidth',1);
        
        title("n=" + n_list(ni),'FontSize',14);
        ylim('padded');
        xlim(XLIM);
        xlabel("#moduli",'FontSize',14);
        set(gca,'FontSize',14);
    end
    lgd = legend;
    lgd.Layout.Tile = 'east'; %'south';
    lgd.NumColumns = 1; %length(func_list);
    lgd.FontSize = 14;
    pattern = "time_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    title(t, env, 'FontSize',14);
    ylabel(t, "TFLOPS",'FontSize',14);
    t.TileSpacing = "tight";
    t.Padding = "compact";
    
    if saveflag
        pattern = "(.*?)_2025";
        match = regexp(filename, pattern, 'tokens');
        figname = match{1}{1};
        savefig(fig,savedir+figname);
        exportgraphics(fig,savedir+figname + ".png",'Resolution',600);
    end
end

end

%%
function m = mark(i,j)
markers = {"-", "--", "-d", "-+", "-o", "-s", "-x", "-p", "-h", "-^", "-v", "->", "-<"};
colors = {"k", "m", "r", "b", "g"};
m = markers{i} + colors(j);
end
