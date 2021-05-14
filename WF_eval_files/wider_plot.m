function wider_plot(set_list, dir_ext, out_dir, prefix_dir)

method_list = dir(dir_ext);
model_num = size(method_list,1) - 2;
model_name = cell(model_num,1);

for i = 3:size(method_list,1)
    model_name{i-2} = method_list(i).name;
end

propose = cell(model_num,1);
recall = cell(model_num,1);
name_list = cell(model_num,1);
ap_list = zeros(3,1);
for i = 1:size(set_list,1)
    load(sprintf('%s/%s/wider_pr_info_%s_%s.mat',dir_ext, prefix_dir, prefix_dir, set_list{i}));
    propose{i} = pr_cruve(:,2);
    recall{i} = pr_cruve(:,1);
    ap = VOCap(propose{i},recall{i});
    ap_list(i) = ap;
end

mAPOut = fopen(out_dir, 'w');
for i = 1:size(ap_list, 1)
    fprintf(mAPOut, "%s,%f\n", char(set_list(i)), ap_list(i));
end
fclose(mAPOut);

%{
for i = 1:size(set_list,1)
    propose = cell(model_num,1);
    recall = cell(model_num,1);
    name_list = cell(model_num,1);
    ap_list = zeros(model_num,1);
    for j = 1:model_num
        load(sprintf('%s/%s/wider_pr_info_%s_%s.mat',dir_ext, model_name{j}, model_name{j}, set_list{i}));
        propose{j} = pr_cruve(:,2);
        recall{j} = pr_cruve(:,1);
        ap = VOCap(propose{j},recall{j});
        ap_list(j) = ap;
        fprintf("ap: %s\n", ap);
        ap = num2str(ap);
        fprintf("ap: %d\n", ap);
        if length(ap) < 5
            name_list{j} = [legend_name '-' ap];
        else
            name_list{j} = [legend_name '-' ap(1:5)];
        end       
    end
    [~,index] = sort(ap_list,'descend');
    propose = propose(index);
    recall = recall(index);
    name_list = name_list(index);
    %plot_pr(propose, recall, name_list, seting_class, set_list{i},dateset_class);
end
%}

