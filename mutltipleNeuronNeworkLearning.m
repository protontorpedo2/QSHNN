e=exp(1);% bottom of natural logarithm

w_11 = [0.03,0.1,0.01,0.06];% initial weight quaternion
w_12 = [0.03,0.1,0.01,0.06];% initial weight quaternion
w_21 = [0.03,0.1,0.01,0.06];% initial weight quaternion
w_22 = [0.03,0.1,0.01,0.06];% initial weight quaternion
w=[w_11 w_12;w_21 w_22];

q_0=[1,2,4,3,1,2,4,3];% initial value of neural network
u = 0.05;% learning rate constant
q = q_0;
d = [0.06,0.12,0.34,0.5,0.06,0.12,0.34,0.5];
MSE = 1;% mean square error
period = 0;% training loop times
data_loss = [];
data_1 = [];data_2 = [];data_3 = [];data_4 = [];
data_5 = [];data_6 = [];data_7 = [];data_8 = [];

while MSE > 0.001
    %mean square error between result and desire value
    MSE = sqrt(sum((d-q).^2))
    %disp(MSE);
    
    %visualize recorded data
    data_loss = [data_loss,MSE];
    data_1 = [data_1;q(1)];
    data_2 = [data_2;q(2)];
    data_3 = [data_3;q(3)];
    data_4 = [data_4;q(4)];
    data_5 = [data_5;q(5)];
    data_6 = [data_6;q(6)];
    data_7 = [data_7;q(7)];
    data_8 = [data_8;q(8)];
    
    delta = d-q;
    delta1 = delta(1,1:4);
    delta2 = delta(1,5:8);
    
    % iterative updating
    w_11 = (w_11'+u.*(matrix(delta1)*([q(1),-q(2),-q(3),-q(4)]')))';
    w_12 = (w_12'+u.*(matrix(delta1)*([q(5),-q(6),-q(7),-q(8)]')))';
    w_21 = (w_21'+u.*(matrix(delta2)*([q(1),-q(2),-q(3),-q(4)]')))';
    w_22 = (w_22'+u.*(matrix(delta2)*([q(5),-q(6),-q(7),-q(8)]')))';
    %operation result of updating network
    q = nt(q_0,w_11,w_12,w_21,w_22);
    
    %training period calculator
    period = period+1
end

plot(data_loss,'linewidth',1.5);
grid on;
%plot(data_2(30:end));
figure;
plot3(data_4(2:end),data_5(2:end),data_6(2:end),'linewidth',1.5);
grid on;
figure;
plot3(data_6(2:end),data_7(2:end),data_8(2:end),'linewidth',1.5);
grid on;
figure;
plot3(data_5(2:end),data_7(2:end),data_8(2:end),'linewidth',1.5);
grid on;

% operation result of neural network
function p = nt(q_0,w_11,w_12,w_21,w_22)
    e=exp(1);% bottom of natural logarithm
    g = @(x)(e^(x)-e^(-x))/e^(-x)+e^x;% activation function applied
    f = @(t,x)[w_11(1)*g(x(1))-w_11(2)*g(x(2))-w_11(3)*g(x(3))-w_11(4)*g(x(4))+1+ w_12(1)*g(x(5))-w_12(2)*g(x(6))-w_12(3)*g(x(7))-w_12(4)*g(x(8))+1; 
               w_11(2)*g(x(1))+w_11(1)*g(x(2))-w_11(4)*g(x(3))+w_11(3)*g(x(4))+2+ w_12(1)*g(x(5))-w_12(2)*g(x(6))-w_12(3)*g(x(7))-w_12(4)*g(x(8))+1;
               w_11(3)*g(x(1))+w_11(4)*g(x(2))+w_11(1)*g(x(3))-w_11(2)*g(x(4))+3+ w_12(3)*g(x(5))+w_12(4)*g(x(6))+w_12(1)*g(x(7))-w_12(2)*g(x(8))+3;
               w_11(4)*g(x(1))-w_11(3)*g(x(2))+w_11(2)*g(x(3))+w_11(1)*g(x(4))+2+ w_12(4)*g(x(5))-w_12(3)*g(x(6))+w_12(2)*g(x(7))+w_12(1)*g(x(8))+2;
               w_21(1)*g(x(1))-w_21(2)*g(x(2))-w_21(3)*g(x(3))-w_21(4)*g(x(4))+1+ w_22(1)*g(x(5))-w_22(2)*g(x(6))-w_22(3)*g(x(7))-w_22(4)*g(x(8))+1; 
               w_21(2)*g(x(1))+w_21(1)*g(x(2))-w_21(4)*g(x(3))+w_21(3)*g(x(4))+2+ w_22(1)*g(x(5))-w_22(2)*g(x(6))-w_22(3)*g(x(7))-w_22(4)*g(x(8))+1;
               w_21(3)*g(x(1))+w_21(4)*g(x(2))+w_21(1)*g(x(3))-w_21(2)*g(x(4))+3+ w_22(3)*g(x(5))+w_22(4)*g(x(6))+w_22(1)*g(x(7))-w_22(2)*g(x(8))+3;
               w_21(4)*g(x(1))-w_21(3)*g(x(2))+w_21(2)*g(x(3))+w_21(1)*g(x(4))+2+ w_22(4)*g(x(5))-w_22(3)*g(x(6))+w_22(2)*g(x(7))+w_22(1)*g(x(8))+2];
    opts = odeset('RelTol',1e-100,'AbsTol',1e-100);
    [t,y] = ode45(f,[0,5],q_0,opts);
    p = [y(end-2,1) y(end-2,2) y(end-2,3) y(end-2,4) y(end-2,5) y(end-2,6) y(end-2,7) y(end-2,8)];
end

% transfer the algorithm form of quaternion
function W = matrix(w)
   W = [w(1) -w(2) -w(3) -w(4); 
        w(2) w(1) -w(4) w(3);
        w(3) w(4) w(1) -w(2);
        w(4) -w(3) w(2) w(1)];% Matrix representation
end
