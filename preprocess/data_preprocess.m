clear 
close all;

%% setting
num_frame = 12000;  % 每秒150frame, 一共80s
sub_channels = 2048;  % 每frame有2048个子载波
num_antennas = 4;  % 4天线
ant_index = 1;
Rx = 'c';  % 场景中一共有三个AP (a,b,c)

%% 数据加载
Path = 'D:\课程\无线物联网\S2\';
save_Path = 'D:\课程\无线物联网\code\XP\';

File = dir(fullfile(Path,'*c_*.mat'));  % 显示文件夹下所有符合*c_*文件的完整信息
FileNames = {File.name}';            % 提取符合后缀名为.mat的所有文件的文件名，转换为n行1列
[~,ind] = natsort(FileNames);
FileNames = FileNames(ind);
Length_Names = size(FileNames,1);    % 获取所提取数据文件的个数

for z = 1 : Length_Names   
    disp(FileNames{z,1});
    K_Trace = strcat(Path, FileNames{z,1});
    eval(['load(K_Trace)',';']);  %直接导入了相应的变量
   
    RawCSI = csi(:,:,ant_index);  %提取第一个雷达的数据: T * sub_channels
    clear('csi');

    %scatter(real(RawCSI(1,:)),imag(RawCSI(1,:)),[],'.'); 

    csi_phase_raw = angle(RawCSI);
    csi_mag_raw = abs(RawCSI);

    %% Hampel Filter
    window_size = 7; % 窗口大小，可根据需要调整
    num_dev = 2; % 定义异常值的标准偏差范围
 
    % 对每一个子载波进行处理
    csi_mag = zeros(num_frame, sub_channels);
    for k = 1:sub_channels
        % 使用hampel函数进行滤波
        filtered_data = hampel(csi_mag_raw(:, k), window_size, num_dev);
        
%         figure;
%         subplot(2, 1, 1);
%         plot(csi_mag_raw(:, k),'r');
%         hold on
%         subplot(2, 1, 2);
%         plot(filtered_data,'b');

        csi_mag(:, k) = filtered_data;
    end  

    %% Phase Proprocess
    b = sum(csi_phase_raw,2) / sub_channels;
    csi_phase = zeros(num_frame, sub_channels);
    csi_phase(:,1) = csi_phase_raw(:,1);
    for i = 2:sub_channels
        a = (csi_phase_raw(:,i) - csi_phase_raw(:,1)) / (i-1);
        csi_phase(:,i) = csi_phase_raw(:,i) - a*i - b;
    end

    %% 数据保存
    smap_name = FileNames{z,1};
    save(strcat(save_Path,smap_name),"csi_mag","csi_phase");

end