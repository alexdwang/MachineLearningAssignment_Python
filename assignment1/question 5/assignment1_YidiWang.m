T = readtable('SkillCraft1_Dataset.csv');
format long;
GapBetweenPACs = T(:,13);
c1 = table2cell(GapBetweenPACs);
m1 = cell2mat(c1);

figure;
histogram(m1);
title('GapBetweenPACs','fontsize',16);

ActionsInPAC = T(:,15);
c2 = table2cell(ActionsInPAC);
m2 = cell2mat(c2);

figure;
histogram(m2);
title('ActionsInPAC','fontsize',16);

SelectByHotkeys = varfun(@str2double,T(:,7));
c3_1 = table2cell(SelectByHotkeys);
m3_1 = cell2mat(c3_1);
AssignToHotkeys = varfun(@str2double,T(:,8));
c3_2 = table2cell(AssignToHotkeys);
m3_2 = cell2mat(c3_2);

figure;
scatter(m3_1,m3_2);
title('SelectByHotkeys vs.AssignToHotkeys','fontsize',16);

NumberOfPACs = T(:,12);
c3_3 = table2cell(NumberOfPACs);
m3_3 = cell2mat(c3_1);

m3_4 = m1;

figure;
scatter(m3_3,m3_4);
title(' NumberOfPACs vs. GapBetweenPACs','fontsize',16);

subset = T(:,6:20);
c_subset = table2cell(subset);
c_subset(:,2) = num2cell(cellfun(@str2double, c_subset(:,2)));
c_subset(:,3) = num2cell(cellfun(@str2double, c_subset(:,3)));
c_subset(:,5) = num2cell(cellfun(@str2double, c_subset(:,5)));
c_subset(:,6) = num2cell(cellfun(@str2double, c_subset(:,6)));
c_subset(:,12) = num2cell(cellfun(@str2double, c_subset(:,12)));
c_subset(:,14) = num2cell(cellfun(@str2double, c_subset(:,14)));
c_subset(:,15) = num2cell(cellfun(@str2double, c_subset(:,15)));
m_subset = cell2mat(c_subset);

PCCs = zeros(15);
for i=1:15
    for j=i:15
        PCC = corr(m_subset(:,i),m_subset(:,j));
        PCCs(i,j) = PCC;
        PCCs(j,i) = PCC;
        PCCs(i,i) = 0;
    end
end

save('Pearson_Correlation_Coefficient_Matrix.mat','PCCs');

minimum = min(PCCs);
maximum = max(PCCs);

minimum_value = min(minimum)
maximum_value = max(maximum)

[i1,j1] = find(PCCs == minimum_value);
[i2,j2] = find(PCCs == maximum_value);

figure;
scatter(m_subset(:,i1(1)),m_subset(:,j1(1)));
title('Minimum PCC pairs','fontsize',16);

figure;
scatter(m_subset(:,i2(1)),m_subset(:,j2(1)));
title('Maximum PCC pairs','fontsize',16);
