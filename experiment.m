clear;
e = 3;%bottom of natural logarithm
n=4;%number of neurons

q0 = [0,0.2,0.4,0.3];% initial value of neurons
q0 = [2,5,6,8];
q_0 = [];
for i=1:n
    q_0 = [q_0,q0]; %#ok<AGROW>
end 

w0 = [0,0.01,0.01,0.06];%initial weight values
w = ones(n^2,4);
for i=1:n^2
    w(i,:) = w0.*(i/(n^2));%randow value generator
end

W=[];%adjacent matrix
L = [];
for i = 1:n
    for j = 1:n
        L =  [L,matrix(w(n*(i-1)+j,:))];
    end
    W = [W;L];
    L = [];
end

u = 0.03;% learning rate
u = 0.03;
q = q_0;

d0 = [0.5,-0.2,0.4,0.6];
d = [];
for i=1:n
    % if i == 2
    %     d = [d,d0*2];
    % elseif i ==3
    %     d = [d,d0*3];
    % else
    d = [d,i.*d0]; %#ok<AGROW> %training desire
    % end
end 

MSE = 1;% mean square error
period = 0;% training loop times
data_loss = [];
data_1 = [];data_2 = [];data_3 = [];data_4 = [];
data_5 = [];data_6 = [];data_7 = [];data_8 = [];
data_9 = [];data_10 = [];data_11 = [];data_12 = [];
data_13 = [];data_14 = [];data_15 = [];data_16 = [];

syms  t g h
x = sym('x',[n*4,1]);
g = (e^(t)-e^(-t))/(e^(t)+e^t); % hyperbolic activation function

for i = 1:n*4
    x(i) = subs(g,x(i));
end

delta = d-q;
 for i = 1:n
        a = (i-1)*4;
        for j = 1:n
             b = (j-1)*4;
            if i ~= j
                %update W_ij, neuron_j influence neuron_i through link W_ij.
                W(a+1:a+4,b+1) = (u.*(matrix(delta(a+1:a+4))*([q(b+1),-q(b+2),-q(b+3),-q(b+4)].'))).';
                W(a+1:a+4,b+1) = W(a+1:a+4,b+1) ./Modu(W(a+1:a+4,b+1)); %#ok<SAGROW>
                W(a+1:a+4,b+1:b+4) = matrix(W(a+1:a+4,b+1));
                %W(b+1:b+4,a+1:a+4) =  W(a+1:a+4,b+1:b+4);
            else
                W(a+1:a+4,b+1) = (u.*(matrix(delta(a+1:a+4))*([0,-q(b+2),-q(b+3),-q(b+4)].'))).';
                W(a+1:a+4,b+1) = W(a+1:a+4,b+1) ./Modu(W(a+1:a+4,b+1)); %#ok<SAGROW>
                W(a+1:a+4,b+1:b+4) = matrix(W(a+1:a+4,b+1));
            end
        end
 end

while MSE > 0.003
    
    MSE = sqrt(sum((d-q).^2));%mean square error between result and desire value
    disp('MSE:');
    disp(MSE);
     %visualize recorded data
    data_loss = [data_loss,MSE]; %#ok<AGROW>

    delta = d-q;%[1,4*n] array, also q and d
    %disp(delta);

    data_1 = [data_1;q(1)]; %#ok<AGROW>
    data_2 = [data_2;q(2)]; %#ok<AGROW>
    data_3 = [data_3;q(3)]; %#ok<AGROW>
    data_4 = [data_4;q(4)];%#ok<AGROW>
    data_5 = [data_5;q(5)];%#ok<AGROW>
    data_6 = [data_6;q(6)];%#ok<AGROW>
    data_7 = [data_7;q(7)];%#ok<AGROW>
    data_8 = [data_8;q(8)];%#ok<AGROW>
    data_9 = [data_9;q(9)];%#ok<AGROW>
    data_10 = [data_10;q(10)];%#ok<AGROW>
    data_11 = [data_11;q(11)];%#ok<AGROW>
    data_12 = [data_12;q(12)];%#ok<AGROW>
    data_13 = [data_13;q(13)];%#ok<AGROW>
    data_14 = [data_14;q(14)];%#ok<AGROW>
    data_15 = [data_15;q(15)];%#ok<AGROW>
    data_16 = [data_16;q(16)];%#ok<AGROW>
    
    % iterative update weight as gradient descent steps
    for i = 1:n
        a = (i-1)*4;
        for j = 1:n
            if i ~= j    
                %update W_ij, neuron_j influence neuron_i through link W_ij.
                b = (j-1)*4;
                W(a+1:a+4,b+1) = (W(a+1:a+4,b+1)+u.*(matrix(delta(a+1:a+4))*([q(b+1),-q(b+2),-q(b+3),-q(b+4)].'))).'; %#ok<SAGROW>
                W(a+1:a+4,b+1) = W(a+1:a+4,b+1) ./Modu(W(a+1:a+4,b+1)); %#ok<SAGROW>
                W(a+1:a+4,b+1:b+4) = matrix(W(a+1:a+4,b+1));
                %W(b+1:b+4,a+1:a+4) =  W(a+1:a+4,b+1:b+4);
            else
                b = (j-1)*4;
                W(a+1:a+4,b+1) = (W(a+1:a+4,b+1)+u.*(matrix(delta(a+1:a+4))*([0,-q(b+2),-q(b+3),-q(b+4)].'))).'; %#ok<SAGROW>
                W(a+1:a+4,b+1) = W(a+1:a+4,b+1) ./Modu(W(a+1:a+4,b+1)); %#ok<SAGROW>
                W(a+1:a+4,b+1:b+4) = matrix(W(a+1:a+4,b+1));
            end
        end
    end
    %disp(W);

    h = W*x-0.01*x;%\gamma = 0.01
    vpa(h,5);
    f = char(h);
    f3 = strcat('@(t,x)',f);%combine two char rings
    
    for i = 4*n:-1:1
        num = num2str(i);
        str = strcat('x',num);
        str2 = strcat('x(',num,')');
        f3 = strrep(f3,str,str2);
        %f3 = strrep(f3,'x12','x(12)');
    end
    %disp(f3);
    f3 = str2func(f3);
    q = networkOperate(q_0,f3);%new result of updated network
    period = period+1;%training period calculator
    disp('period:');
    disp(period);
    % disp('Differential solver succeed!');
    % disp(q);
end

figure;
plot(data_loss(2:end),'linewidth',2);
title("Loss Function Curve over Training Iterations");
legend("Loss Function Curve, μ=0.03");
xlabel("Epoch Count");
ylabel("Mean Square Error (MSE)");
grid on;

figure;
hold on;
x = 2:length(data_loss);
plot(x,data_1(2:end),'Color',[0.4660 0.6740 0.1880],'LineWidth',2);
plot(x,data_2(2:end),'Color',[0.8500 0.3250 0.0980],'LineWidth',1.2);
plot(x,data_3(2:end),'Color',[0.8500 0.3250 0.0980],'LineWidth',1.2);
plot(x,data_4(2:end),'Color',[0.8500 0.3250 0.0980],'LineWidth',1.2);
title("Traing Curve over Training Iterations Q1");
legend("Scalar Component","Imaginary Component x","Imaginary Component y","Imaginary Component z");
xlabel("Epoch Count");
ylabel("Network Convergency State");
hold off;
grid on;

figure;
hold on;
x = 2:length(data_loss);
plot(x,data_5(2:end),'Color',[0.4660 0.6740 0.1880],'LineWidth',2);
plot(x,data_6(2:end),'Color',[0.8500 0.3250 0.0980],'LineWidth',1.2);
plot(x,data_7(2:end),'Color',[0.8500 0.3250 0.0980],'LineWidth',1.2);
plot(x,data_8(2:end),'Color',[0.8500 0.3250 0.0980],'LineWidth',1.2);
title("Traing Curve over Training Iterations Q2");
legend("Scalar Component","Imaginary Component x","Imaginary Component y","Imaginary Component z");
xlabel("Epoch Count");
ylabel("Network Convergency State");
hold off;
grid on;

figure;
hold on;
x = 2:length(data_loss);
plot(x,data_9(2:end),'Color',[0.4660 0.6740 0.1880],'LineWidth',2);
plot(x,data_10(2:end),'Color',[0.8500 0.3250 0.0980],'LineWidth',1.2);
plot(x,data_11(2:end),'Color',[0.8500 0.3250 0.0980],'LineWidth',1.2);
plot(x,data_12(2:end),'Color',[0.8500 0.3250 0.0980],'LineWidth',1.2);
title("Traing Curve over Training Iterations Q3");
legend("Scalar Component","Imaginary Component x","Imaginary Component y","Imaginary Component z");
xlabel("Epoch Count");
ylabel("Network Convergency State");
hold off;
grid on;

figure;
hold on;
x = 2:length(data_loss);
plot(x,data_13(2:end),'Color',[0.4660 0.6740 0.1880],'LineWidth',2);
plot(x,data_14(2:end),'Color',[0.8500 0.3250 0.0980],'LineWidth',1.2);
plot(x,data_15(2:end),'Color',[0.8500 0.3250 0.0980],'LineWidth',1.2);
plot(x,data_16(2:end),'Color',[0.8500 0.3250 0.0980],'LineWidth',1.2);
title("Traing Curve over Training Iterations Q4");
legend("Scalar Component","Imaginary Component x","Imaginary Component y","Imaginary Component z");
xlabel("Epoch Count");
ylabel("Network Convergency State");
hold off;
grid on;
clear;

% operation result of neural network
function q = networkOperate(q_0,f3)
    opts = odeset('RelTol',1e-10,'AbsTol',1e-100,'MaxStep',0.05);
    [~,y] = ode45(f3,[0,5],q_0,opts);
    q = y(end,:);
end

% Matrix representation
function Q = matrix(q)
   Q = [q(1) -q(2) -q(3) -q(4); 
           q(2) q(1) -q(4) q(3);
           q(3) q(4) q(1) -q(2);
           q(4) -q(3) q(2) q(1)];
end

%Quaternion Multiplication
function Q = multiplication(p,q)
    Q = (matrix(p)*q.').';
end

%L2 Norm Calculation
function m = Modu(q)
    m = sqrt(sum(q.^(2)));
end
