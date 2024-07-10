clc; clear;
w = [0.03,0.1,0.01,0.06];% initial weight quaternion
q_0=[-1,2,-4,3];% initial value of neural network
u = 0.03;% learning rate constant
q = q_0;% output value of neural network
d = [-0.6,1.2,-0.34,0.5];% desired learning target

 global e;
 e = exp(1);% bottom of natural logarithm
 global syms x1 x2 x3 x4 x;
 global syms w1 w2 w3 w4;
 global g;
 g = (e^(x)-e^(-x))/(e^(-x)+e^x);% activation function applied
 global f;
 f = matrix(w1,w2,w3,w4)*[subs(g,x,x1),subs(g,x,x2),subs(g,x,x3),subs(g,x,x4)].';
 global opts;
 opts = odeset('RelTol',1e-20,'AbsTol',1e-20);

MSE = 1;% mean square error
period = 0;% training loop times
data_loss = [];
data_1 = [];data_2 = [];data_3 = [];data_4 = [];

while MSE > 0.01 && period<100
    %mean square error between result and desire value
    MSE = sqrt(sum((d-q).^2))
    %disp(MSE);
    
    %visualize recorded data
    data_loss = [data_loss,MSE];
    data_1 = [data_1;q(1)];
    data_2 = [data_2;q(2)];
    data_3 = [data_3;q(3)];
    data_4 = [data_4;q(4)];
    
    % iterative updating
    w = (w'+u.*(matrixConvention(d-q)*([q(1),-q(2),-q(3),-q(4)]')))'
    matrixConvention(d-q)*([q(1),-q(2),-q(3),-q(4)]')
    %operation result of updating network
    q = nt(q_0,w);
    
    %training period calculator
    period = period+1
end

plot(data_loss,'linewidth',1.5);
grid on;
% %plot(data_2(30:end));
figure;
plot3(data_1(2:end),data_2(2:end),data_3(2:end),'linewidth',1.5);
% grid on;
% figure;
% plot3(data_1(2:end),data_2(2:end),data_4(2:end),'linewidth',1.5);
% grid on;
% figure;
% plot3(data_2(2:end),data_3(2:end),data_4(2:end),'linewidth',1.5);
% grid on;

% operation result of neural network
function p = nt(q_0,w)
    global f;
    global w1 w2 w3 w4;
    global x1 x2 x3 x4;
    global opts;
    fs = subs(f,{w1,w2,w3,w4},w);
    fss = matlabFunction([fs(1);fs(2);fs(3);fs(4)],'Vars',[x1,x2,x3,x4]);
    fsss = @(t,x)fss(x(1),x(2),x(3),x(4));
    [t,y] = ode45(fsss,[0,5],q_0,opts);
    p = [y(end-2,1) y(end-2,2) y(end-2,3) y(end-2,4)];
end

% transform symbolic variable to matrix convention 
function W = matrix(w1,w2,w3,w4)
   W = [w1 -w2 -w3 -w4; 
        w2 w1 -w4 w3;
        w3 w4 w1 -w2;
        w4 -w3 w2 w1];
end

%matrix convention of quaternion
function W = matrixConvention(w)
    W = [w(1) -w(2) -w(3) -w(4); 
         w(2) w(1) -w(4) w(3);
         w(3) w(4) w(1) -w(2);
         w(4) -w(3) w(2) w(1)];
end
