function plot_timebreakdown(saveflag,savedir)

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
    opts.SelectedVariableNames = 10:13;
    times = readmatrix(filename,opts);
    moduli_min = 2;
    moduli_max = 15;
    XLIM = [4 12];
    
    labels = replace(string(opts.SelectedVariableNames), "_", "\_");
    
    fig = figure;
    fig.Position(3) = 1000;
    fig.Position(4) = 350;
    t = tiledlayout(2,length(n_list));
    for ni = 1:length(n_list)
        nexttile(ni); hold on; grid on;
        flag = false(size(n));
        for m = XLIM(1):XLIM(2)
            flag = flag | contains(func,"OS2-fast-" + m);
        end
        flag = flag & (n == n_list(ni));
        OS2_fast = times(flag,:);
        bar(OS2_fast./sum(OS2_fast,2)*100,'stacked');
        ylim([0 100]);
        xticks(1:(XLIM(2)-XLIM(1)+1));
        xl = string(XLIM(1):XLIM(2));
        xl(2:2:end) = "";
        xticklabels(xl);
        yticks(0:25:100);
        if ni>1
            yticklabels("");
        else
            ylabel("%",'FontSize',14);
        end
        title("fast n=" + n_list(ni));
        xlabel("#moduli",'FontSize',14);
        set(gca,'FontSize',14);
        
        nexttile(ni + length(n_list)); hold on; grid on;
        flag = false(size(n));
        for m = XLIM(1):XLIM(2)
            flag = flag | contains(func,"OS2-accu-" + m);
        end
        flag = flag & (n == n_list(ni));
        OS2_accu = times(flag,:);
        bar(OS2_accu./sum(OS2_accu,2)*100,'stacked');
        ylim([0 100]);
        xticks(1:(XLIM(2)-XLIM(1)+1));
        xl = string(XLIM(1):XLIM(2));
        xl(2:2:end) = "";
        xticklabels(xl);
        yticks(0:25:100);
        if ni>1
            yticklabels("");
        else
            ylabel("%",'FontSize',14);
        end
        title("accu n=" + n_list(ni));
        xlabel("#moduli",'FontSize',14);
        set(gca,'FontSize',14);
    end
    lgd = legend(labels);
    lgd.Layout.Tile = 'east';%'south';
    lgd.NumColumns = 1;%length(func_list);
    lgd.FontSize = 14;
    pattern = "time_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    title(t, env, 'FontSize',14);
    t.TileSpacing = "tight";
    t.Padding = "compact";
    
    if saveflag
        pattern = "(.*?)_2025";
        match = regexp(filename, pattern, 'tokens');
        figname = match{1}{1};
        figname = replace(figname, "time", "timebreakdown");
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
    opts.SelectedVariableNames = 10:13;
    times = readmatrix(filename,opts);
    moduli_min = 2;
    moduli_max = 20;
    XLIM = [8 moduli_max];
    
    labels = replace(string(opts.SelectedVariableNames), "_", "\_");
    
    fig = figure;
    fig.Position(3) = 1000;
    fig.Position(4) = 350;
    t = tiledlayout(2,length(n_list));
    for ni = 1:length(n_list)
        nexttile(ni); hold on; grid on;
        flag = false(size(n));
        for m = XLIM(1):XLIM(2)
            flag = flag | contains(func,"OS2-fast-" + m);
        end
        flag = flag & (n == n_list(ni));
        OS2_fast = times(flag,:);
        bar(OS2_fast./sum(OS2_fast,2)*100,'stacked');
        ylim([0 100]);
        xticks(1:(XLIM(2)-XLIM(1)+1));
        xl = string(XLIM(1):XLIM(2));
        xl(2:2:end) = "";
        xticklabels(xl);
        yticks(0:25:100);
        if ni>1
            yticklabels("");
        else
            ylabel("%",'FontSize',14);
        end
        title("fast n=" + n_list(ni));
        xlabel("#moduli",'FontSize',14);
        set(gca,'FontSize',14);
        
        nexttile(ni + length(n_list)); hold on; grid on;
        flag = false(size(n));
        for m = XLIM(1):XLIM(2)
            flag = flag | contains(func,"OS2-accu-" + m);
        end
        flag = flag & (n == n_list(ni));
        OS2_accu = times(flag,:);
        bar(OS2_accu./sum(OS2_accu,2)*100,'stacked');
        ylim([0 100]);
        xticks(1:(XLIM(2)-XLIM(1)+1));
        xl = string(XLIM(1):XLIM(2));
        xl(2:2:end) = "";
        xticklabels(xl);
        yticks(0:25:100);
        if ni>1
            yticklabels("");
        else
            ylabel("%",'FontSize',14);
        end
        title("accu n=" + n_list(ni));
        xlabel("#moduli",'FontSize',14);
        set(gca,'FontSize',14);
    end
    lgd = legend(labels);
    lgd.Layout.Tile = 'east';%'south';
    lgd.NumColumns = 1;%length(func_list);
    lgd.FontSize = 14;
    pattern = "time_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    title(t, env, 'FontSize',14);
    t.TileSpacing = "tight";
    t.Padding = "compact";
    
    if saveflag
        pattern = "(.*?)_2025";
        match = regexp(filename, pattern, 'tokens');
        figname = match{1}{1};
        figname = replace(figname, "time", "timebreakdown");
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
    opts.SelectedVariableNames = 10:13;
    times = readmatrix(filename,opts);
    moduli_min = 2;
    moduli_max = 20;
    XLIM = [8 moduli_max];
    
    labels = replace(string(opts.SelectedVariableNames), "_", "\_");
    
    fig = figure;
    fig.Position(3) = 1000;
    fig.Position(4) = 350;
    t = tiledlayout(2,length(n_list));
    for ni = 1:length(n_list)
        nexttile(ni); hold on; grid on;
        flag = false(size(n));
        for m = XLIM(1):XLIM(2)
            flag = flag | contains(func,"OS2-fast-" + m);
        end
        flag = flag & (n == n_list(ni));
        OS2_fast = times(flag,:);
        bar(OS2_fast./sum(OS2_fast,2)*100,'stacked');
        ylim([0 100]);
        xticks(1:(XLIM(2)-XLIM(1)+1));
        xl = string(XLIM(1):XLIM(2));
        xl(2:2:end) = "";
        xticklabels(xl);
        yticks(0:25:100);
        if ni>1
            yticklabels("");
        else
            ylabel("%",'FontSize',14);
        end
        title("fast n=" + n_list(ni));
        xlabel("#moduli",'FontSize',14);
        set(gca,'FontSize',14);
        
        nexttile(ni + length(n_list)); hold on; grid on;
        flag = false(size(n));
        for m = XLIM(1):XLIM(2)
            flag = flag | contains(func,"OS2-accu-" + m);
        end
        flag = flag & (n == n_list(ni));
        OS2_accu = times(flag,:);
        bar(OS2_accu./sum(OS2_accu,2)*100,'stacked');
        ylim([0 100]);
        xticks(1:(XLIM(2)-XLIM(1)+1));
        xl = string(XLIM(1):XLIM(2));
        xl(2:2:end) = "";
        xticklabels(xl);
        yticks(0:25:100);
        if ni>1
            yticklabels("");
        else
            ylabel("%",'FontSize',14);
        end
        title("accu n=" + n_list(ni));
        xlabel("#moduli",'FontSize',14);
        set(gca,'FontSize',14);
    end
    lgd = legend(labels);
    lgd.Layout.Tile = 'east';%'south';
    lgd.NumColumns = 1;%length(func_list);
    lgd.FontSize = 14;
    pattern = "time_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    title(t, env, 'FontSize',14);
    t.TileSpacing = "tight";
    t.Padding = "compact";
    
    if saveflag
        pattern = "(.*?)_2025";
        match = regexp(filename, pattern, 'tokens');
        figname = match{1}{1};
        figname = replace(figname, "time", "timebreakdown");
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
    opts.SelectedVariableNames = 10:13;
    times = readmatrix(filename,opts);
    moduli_min = 2;
    moduli_max = 15;
    XLIM = [4 12];
    
    labels = replace(string(opts.SelectedVariableNames), "_", "\_");
    
    fig = figure;
    fig.Position(3) = 1000;
    fig.Position(4) = 350;
    t = tiledlayout(2,length(n_list));
    for ni = 1:length(n_list)
        nexttile(ni); hold on; grid on;
        flag = false(size(n));
        for m = XLIM(1):XLIM(2)
            flag = flag | contains(func,"OS2-fast-" + m);
        end
        flag = flag & (n == n_list(ni));
        OS2_fast = times(flag,:);
        bar(OS2_fast./sum(OS2_fast,2)*100,'stacked');
        ylim([0 100]);
        xticks(1:(XLIM(2)-XLIM(1)+1));
        xl = string(XLIM(1):XLIM(2));
        xl(2:2:end) = "";
        xticklabels(xl);
        yticks(0:25:100);
        if ni>1
            yticklabels("");
        else
            ylabel("%",'FontSize',14);
        end
        title("fast n=" + n_list(ni));
        xlabel("#moduli",'FontSize',14);
        set(gca,'FontSize',14);
        
        nexttile(ni + length(n_list)); hold on; grid on;
        flag = false(size(n));
        for m = XLIM(1):XLIM(2)
            flag = flag | contains(func,"OS2-accu-" + m);
        end
        flag = flag & (n == n_list(ni));
        OS2_accu = times(flag,:);
        bar(OS2_accu./sum(OS2_accu,2)*100,'stacked');
        ylim([0 100]);
        xticks(1:(XLIM(2)-XLIM(1)+1));
        xl = string(XLIM(1):XLIM(2));
        xl(2:2:end) = "";
        xticklabels(xl);
        yticks(0:25:100);
        if ni>1
            yticklabels("");
        else
            ylabel("%",'FontSize',14);
        end
        title("accu n=" + n_list(ni));
        xlabel("#moduli",'FontSize',14);
        set(gca,'FontSize',14);
    end
    lgd = legend(labels);
    lgd.Layout.Tile = 'east';%'south';
    lgd.NumColumns = 1;%length(func_list);
    lgd.FontSize = 14;
    pattern = "time_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    title(t, env, 'FontSize',14);
    t.TileSpacing = "tight";
    t.Padding = "compact";
    
    if saveflag
        pattern = "(.*?)_2025";
        match = regexp(filename, pattern, 'tokens');
        figname = match{1}{1};
        figname = replace(figname, "time", "timebreakdown");
        savefig(fig,savedir+figname);
        exportgraphics(fig,savedir+figname + ".png",'Resolution',600);
    end
end

end

