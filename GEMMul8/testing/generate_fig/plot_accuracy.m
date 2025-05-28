function plot_accuracy(saveflag,savedir)

% close all

if nargin == 0
    saveflag = false;
end

%% files
fileList_f = dir("oz2_results_f_accuracy_*");
filename_f = strings(length(fileList_f),1);
for i=1:length(fileList_f)
    filename_f(i) = string(fileList_f(i).name);
end
fileList_d = dir("oz2_results_d_accuracy_*");
filename_d = strings(length(fileList_d),1);
for i=1:length(fileList_d)
    filename_d(i) = string(fileList_d(i).name);
end
fileList_dfd = dir("oz2_results_dfd_accuracy_*");
filename_dfd = strings(length(fileList_dfd),1);
for i=1:length(fileList_dfd)
    filename_dfd(i) = string(fileList_dfd(i).name);
end

fileList_dff = dir("oz2_results_dff_accuracy_*");
filename_dff = strings(length(fileList_dff),1);
for i=1:length(fileList_dff)
    filename_dff(i) = string(fileList_dff(i).name);
end

%% float
for fn = 1:length(filename_f)
    filename = filename_f(fn);
    opts = detectImportOptions(filename);
    opts.SelectedVariableNames = 3:length(opts.SelectedVariableNames);
    err = readmatrix(filename,opts);
    err = err(2:end,:);
    opts.SelectedVariableNames = 1;
    phi = readmatrix(filename,opts);
    phi = phi(2:end);
    phi_list = unique(phi,'stable');
    opts.SelectedVariableNames = 2;
    func = string(readmatrix(filename,opts));
    func = func(2:end);
    func_list = unique(func,'stable');
    moduli_min = 2;
    moduli_max = moduli_min + size(err,2) - 1;
    xx = moduli_min:moduli_max;
    XLIM = [2 12];

    fig = figure;
    fig.Position(3) = 1000;
    fig.Position(4) = 400;
    t = tiledlayout(1,length(phi_list));
    for p = 1:length(phi_list)
        nexttile; hold on;
        ax = gca;
        ax.XGrid = 'on';
        ax.YGrid = 'on';
        mi = [0,0,0,0];
        ci = [1,2,3,4];
        for fi = 1:length(func_list)
            data = err(phi == phi_list(p) & func == func_list(fi),:);
            idx = (contains(func_list(fi),"SGEMM")&~contains(func_list(fi),"SGEMM-TF32"))*1 + contains(func_list(fi),"SGEMM-TF32")*2 + contains(func_list(fi),"OS2-fast")*3 + contains(func_list(fi),"OS2-accu")*4;
            mi(idx) = mi(idx)+1;
            fn = replace(func_list(fi),"OS2","OS II");
            plot(xx, data, mark(mi(idx),ci(idx)), 'DisplayName', fn, 'LineWidth',1);
        end

        title("$\phi=" + phi_list(p) + "$",'Interpreter','Latex','FontSize',14);
        ylim('padded');
        yticks(10.^(-20:4:30));
        xlim(XLIM);
        xticks(XLIM(1):2:XLIM(2));
        xlabel("#moduli",'FontSize',14);
        set(gca,'YScale','Log','FontSize',14);
    end
    lgd = legend;
    lgd.Layout.Tile = 'east';
    lgd.NumColumns = 1;
    lgd.FontSize = 11;
    pattern = "accuracy_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    title(t, env, 'FontSize',14);
    ylabel(t, "$\max_{ij} |(AB)_{ij} - C_{ij}|/|(AB)_{ij}|$",'Interpreter','Latex','FontSize',14);
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
    opts.SelectedVariableNames = 3:length(opts.SelectedVariableNames);
    err = readmatrix(filename,opts);
    err = err(2:end,:);
    opts.SelectedVariableNames = 1;
    phi = readmatrix(filename,opts);
    phi = phi(2:end);
    phi_list = unique(phi,'stable');
    opts.SelectedVariableNames = 2;
    func = string(readmatrix(filename,opts));
    func = func(2:end);
    func_list = unique(func,'stable');
    moduli_min = 2;
    moduli_max = moduli_min + size(err,2) - 1;
    xx = moduli_min:moduli_max;
    XLIM = [8 moduli_max];

    fig = figure;
    fig.Position(3) = 1000;
    fig.Position(4) = 350;
    t = tiledlayout(1,length(phi_list));
    for p = 1:length(phi_list)
        nexttile; hold on;
        ax = gca;
        ax.XGrid = 'on';
        ax.YGrid = 'on';
        mi = [0,0,0];
        ci = [1,3,4];
        for fi = 1:length(func_list)
            data = err(phi == phi_list(p) & func == func_list(fi),:);
            idx = contains(func_list(fi),"DGEMM")*1 + contains(func_list(fi),"OS2-fast")*2 + contains(func_list(fi),"OS2-accu")*3;
            mi(idx) = mi(idx)+1;
            fn = replace(func_list(fi),"OS2","OS II");
            plot(xx, data, mark(mi(idx),ci(idx)), 'DisplayName', fn, 'LineWidth',1);
        end

        title("$\phi=" + phi_list(p) + "$",'Interpreter','Latex','FontSize',14);
        ylim('padded');
        yticks(10.^(-20:4:30));
        xlim(XLIM);
        xticks(XLIM(1):2:XLIM(2));
        xlabel("#moduli",'FontSize',14);
        set(gca,'YScale','Log','FontSize',14);
    end
    lgd = legend;
    lgd.Layout.Tile = 'east';
    lgd.NumColumns = 1;
    lgd.FontSize = 11;
    pattern = "accuracy_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    title(t, env, 'FontSize',14);
    ylabel(t, "$\max_{ij} |(AB)_{ij} - C_{ij}|/|(AB)_{ij}|$",'Interpreter','Latex','FontSize',14);
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
    opts.SelectedVariableNames = 3:length(opts.SelectedVariableNames);
    err = readmatrix(filename,opts);
    err = err(2:end,:);
    opts.SelectedVariableNames = 1;
    phi = readmatrix(filename,opts);
    phi = phi(2:end);
    phi_list = unique(phi,'stable');
    opts.SelectedVariableNames = 2;
    func = string(readmatrix(filename,opts));
    func = func(2:end);
    func_list = unique(func,'stable');
    moduli_min = 2;
    moduli_max = moduli_min + size(err,2) - 1;
    xx = moduli_min:moduli_max;
    XLIM = [8 moduli_max];

    fig = figure;
    fig.Position(3) = 1000;
    fig.Position(4) = 350;
    t = tiledlayout(1,length(phi_list));
    for p = 1:length(phi_list)
        nexttile; hold on;
        ax = gca;
        ax.XGrid = 'on';
        ax.YGrid = 'on';
        mi = [0,0,0];
        ci = [1,3,4];
        for fi = 1:length(func_list)
            data = err(phi == phi_list(p) & func == func_list(fi),:);
            idx = contains(func_list(fi),["DGEMM","DSGEMM"])*1 + contains(func_list(fi),"OS2-fast")*2 + contains(func_list(fi),"OS2-accu")*3;
            mi(idx) = mi(idx)+1;
            fn = replace(func_list(fi),"OS2","OS II");
            plot(xx, data, mark(mi(idx),ci(idx)), 'DisplayName', fn, 'LineWidth',1);
        end

        title("$\phi=" + phi_list(p) + "$",'Interpreter','Latex','FontSize',14);
        ylim('padded');
        yticks(10.^(-20:4:30));
        xlim(XLIM);
        xticks(XLIM(1):2:XLIM(2));
        xlabel("#moduli",'FontSize',14);
        set(gca,'YScale','Log','FontSize',14);
    end
    lgd = legend;
    lgd.Layout.Tile = 'east';
    lgd.NumColumns = 1;
    lgd.FontSize = 11;
    pattern = "accuracy_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    title(t, env, 'FontSize',14);
    ylabel(t, "$\max_{ij} |(AB)_{ij} - C_{ij}|/|(AB)_{ij}|$",'Interpreter','Latex','FontSize',14);
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
    opts.SelectedVariableNames = 3:length(opts.SelectedVariableNames);
    err = readmatrix(filename,opts);
    err = err(2:end,:);
    opts.SelectedVariableNames = 1;
    phi = readmatrix(filename,opts);
    phi = phi(2:end);
    phi_list = unique(phi,'stable');
    opts.SelectedVariableNames = 2;
    func = string(readmatrix(filename,opts));
    func = func(2:end);
    func_list = unique(func,'stable');
    moduli_min = 2;
    moduli_max = moduli_min + size(err,2) - 1;
    xx = moduli_min:moduli_max;
    XLIM = [2 12];

    fig = figure;
    fig.Position(3) = 1000;
    fig.Position(4) = 400;
    t = tiledlayout(1,length(phi_list));
    for p = 1:length(phi_list)
        nexttile; hold on;
        ax = gca;
        ax.XGrid = 'on';
        ax.YGrid = 'on';
        mi = [0,0,0,0];
        ci = [1,2,3,4];
        for fi = 1:length(func_list)
            data = err(phi == phi_list(p) & func == func_list(fi),:);
            idx = (contains(func_list(fi),"SGEMM")&~contains(func_list(fi),"SGEMM-TF32"))*1 + contains(func_list(fi),"SGEMM-TF32")*2 + contains(func_list(fi),"OS2-fast")*3 + contains(func_list(fi),"OS2-accu")*4;
            mi(idx) = mi(idx)+1;
            fn = replace(func_list(fi),"OS2","OS II");
            plot(xx, data, mark(mi(idx),ci(idx)), 'DisplayName', fn, 'LineWidth',1);
        end

        title("$\phi=" + phi_list(p) + "$",'Interpreter','Latex','FontSize',14);
        ylim('padded');
        yticks(10.^(-20:4:30));
        xlim(XLIM);
        xticks(XLIM(1):2:XLIM(2));
        xlabel("#moduli",'FontSize',14);
        set(gca,'YScale','Log','FontSize',14);
    end
    lgd = legend;
    lgd.Layout.Tile = 'east';
    lgd.NumColumns = 1;
    lgd.FontSize = 11;
    pattern = "accuracy_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    title(t, env, 'FontSize',14);
    ylabel(t, "$\max_{ij} |(AB)_{ij} - C_{ij}|/|(AB)_{ij}|$",'Interpreter','Latex','FontSize',14);
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
