function [weight_input_hidden,weight_prehidden_hidden,weight_hidden_output]=updata_weight(t,yita,Error,train_data,hidden_state,weight_input_hidden,weight_prehidden_hidden,weight_hidden_output)
%%
weight_hidden_output_temp=weight_hidden_output;
weight_input_hidden_temp=weight_input_hidden;
weight_prehidden_hidden_temp=weight_prehidden_hidden;

%% weight_hidden_output权重更新
for n=1:size(Error,1)
    delta_weight_ho(n,:)=2*Error(n,1).*hidden_state(:,1)';
end
weight_hidden_output_temp=weight_hidden_output_temp-yita*delta_weight_ho;

%% weight_input_hidden权重更新
for n=1:size(Error,1)
    for m=1:size(hidden_state,1)
        delta_weight_ih(:,m)=2*Error(n,1).*weight_hidden_output(n,m)*train_data(:,1);
    end
    weight_input_hidden_temp=weight_input_hidden_temp-yita*delta_weight_ih';
end

%% weight_prehidden_hidden权重更新（t=1时，这个权重不更新）
if (t~=1)
    for n=1:size(Error,1)
        for m=1:size(hidden_state,1)
            delta_weight_hh(m,:)=2*Error(n,1).*weight_hidden_output(n,m)*hidden_state(:,t-1)';
        end
        weight_prehidden_hidden_temp=weight_prehidden_hidden_temp-yita*delta_weight_hh;
    end
end
weight_hidden_output=weight_hidden_output_temp;
weight_input_hidden=weight_input_hidden_temp;
weight_prehidden_hidden=weight_prehidden_hidden_temp;
end