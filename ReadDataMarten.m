% Script for reading .MF4 files and saving them to a .mat file
% Files will be named with the date and time of creation
clear; close all;

dir_raw_files = "eigenvalues_balance_assist_120523";
dir_finalized_files = "finalized_logs";


files = dir(fullfile(dir_raw_files, "*.MF4"));

for i = 1:length(files)
    original_path = dir_raw_files + filesep + files(i).name;
    date = regexprep(files(i).date, ' ', '-');
    date = regexprep(date, ':', '-');
    finalized_path = dir_finalized_files + filesep + date;

    if isfile(finalized_path + ".MF4")
        disp(["File with name ",finalized_path, "already exists."])
        continue
    end

    finalized_mdf = mdfFinalize(original_path, finalized_path + ".MF4");

    % Read and decode data
    can_idx = 8;
    mdf_obj = mdf(finalized_path + ".MF4");
    raw_timetable = read(mdf_obj, can_idx, mdf_obj.ChannelNames{can_idx});
    can_db = canDatabase("dbc_files" + filesep + "Fused_new.dbc");
    msg_timetable = canFDMessageTimetable(raw_timetable, can_db);
    msg_timetable.Time = msg_timetable.Time + mdf_obj.InitialTimestamp + hours(2);
    msg_data = canSignalTimetable(msg_timetable);

    save(finalized_path + ".mat", "msg_data");  

    % Write specific timetables to CSV to analyze in python
    writetimetable(msg_data.WSS_Rear, finalized_path + "-wheelspeed.csv")
    writetimetable(msg_data.LWS_STD, finalized_path + "-steerrate.csv")
    writetimetable(msg_data.IMU_Gyro, finalized_path + "-gyro.csv")
end
  

