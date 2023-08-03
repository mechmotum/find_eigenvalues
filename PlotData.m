clear; close all;

% Load data
dir_finalized_files = "finalized_logs";
file_name = "12-May-2023-10-11-04";
load(dir_finalized_files + filesep + file_name + ".mat");

% Plots
figure
ax_speed = subplot(4,1,1);
plot(msg_data.WSS_Rear.Time, msg_data.WSS_Rear.ws_rear)
ylabel("Velocity [m/s]")

ax_steering_rate = subplot(4,1,2);
plot(msg_data.LWS_STD.Time, msg_data.LWS_STD.LWS_SPEED)
ylabel("Steering rate [deg/s]")

ax_roll_rate = subplot(4,1,3);
plot(msg_data.IMU_Gyro.Time, msg_data.IMU_Gyro.gyro_x)
xlabel("Time")
ylabel("Roll rate [??]")

ax_state = subplot(4,1,4);
hold on;
plot(msg_data.PS_State.Time, msg_data.PS_State.state);
title("State of the Steer-Assist system")
yticks([0 1 2 3 4 5 6]);
yticklabels({'Reset', 'Init', 'Fault', 'Ready', 'Calib', 'Align', 'Run'});
ylim([0 6]);

linkaxes([ax_speed, ax_steering_rate, ax_roll_rate, ax_state], 'x')


% Get segments
% date = "2023-05-12" + " ";
% segments = [
%     [datetime(date + "11:00:33.34355", "TimeZone", "UTC", "Format", 'yyyy-MM-dd HH:mm:ss.SSSSSSSSS'), datetime(date + "11:00:34.7923", "TimeZone", "UTC", "Format", 'yyyy-MM-dd HH:mm:ss.SSSSSSSSS')];
%     [datetime(date + "11:00:45.2428", "TimeZone", "UTC", "Format", 'yyyy-MM-dd HH:mm:ss.SSSSSSSSS'), datetime(date + "11:00:46.1628", "TimeZone", "UTC", "Format", 'yyyy-MM-dd HH:mm:ss.SSSSSSSSS')];
%     [datetime(date + "11:00:59.1335", "TimeZone", "UTC", "Format", 'yyyy-MM-dd HH:mm:ss.SSSSSSSSS'), datetime(date + "11:01:00.0035", "TimeZone", "UTC", "Format", 'yyyy-MM-dd HH:mm:ss.SSSSSSSSS')]
% ];
% 
% % Plot segments
% for i = 1:length(segments)
%     time_range = timerange(segments(i,1), segments(i, 2));
%     wss_rear = msg_data.WSS_Rear(time_range, :);
%     lws_std = msg_data.LWS_STD(time_range, :);
%     imu_gyro = msg_data.IMU_Gyro(time_range, :);
% 
%     figure;
%     ax_speed = subplot(3,1,1);
%     plot(wss_rear.Time, wss_rear.ws_rear)
%     ylabel("Velocity [m/s]")
% 
%     ax_steering_rate = subplot(3,1,2);
%     plot(lws_std.Time, lws_std.LWS_ANGLE)
%     ylabel("Steering rate [deg/s]")
% 
%     ax_roll_rate = subplot(3,1,3);
%     plot(imu_gyro.Time, imu_gyro.gyro_x)
%     xlabel("Time")
%     ylabel("Roll rate [??]")
%     
%     linkaxes([ax_speed, ax_steering_rate, ax_roll_rate], 'x')
% 
% %     Save segments to csv
%     start_time = imu_gyro.Time(1);
%     time = zeros(length(imu_gyro.Time), 1);
%     for j = 1:length(imu_gyro.Time)
%         time(j) = seconds(imu_gyro.Time(j) - start_time);
%     end
%     imu_gyro_time = addvars(imu_gyro, time);
% 
%     finalized_path = dir_finalized_files + filesep + "bas-on-gain-10" + filesep + file_name + "-bas-on-gain-10-speed-18";
% %     writetimetable(wss_rear, finalized_path + "-wheelspeed-" + num2str(i) + ".csv")
% %     writetimetable(lws_std, finalized_path + "-steerrate-" + num2str(i) + ".csv")
%     writetimetable(imu_gyro_time, finalized_path + "-gyro-" + num2str(i) + ".csv")
% end


