module ibstructs

    use iso_c_binding
    implicit none
    integer, parameter:: dp=(kind(0.d0))

    type, bind(c) :: barrel

        real(c_double):: d
        real(c_double):: q
        real(c_double):: s
        real(c_double):: w0
        real(c_double):: l_d
        real(c_double):: l_k
        real(c_double):: l0
        real(c_double):: kf

    end type barrel

    type, bind(c) :: powder

        real(c_double):: om
        real(c_double):: ro
        real(c_double):: f_powd
        real(c_double):: ti
        real(c_double):: jk
        real(c_double):: alpha
        real(c_double):: theta
        real(c_double):: zk
        real(c_double):: kappa1
        real(c_double):: lambda1
        real(c_double):: mu1
        real(c_double):: kappa2
        real(c_double):: lambda2
        real(c_double):: mu2
        real(c_double):: gamma_f
        real(c_double):: gamma_jk

    end type powder

    type, bind(c) :: igniter

        real(c_double):: fs
        real(c_double):: num
        real(c_double):: denum

    end type igniter

    type, bind(c) :: ibproblem

        ! Исходные данные
        type(barrel):: artsystem
        real(c_double):: p0
        real(c_double):: pv
        real(c_double):: ig_mass
        real(c_double):: t0
        ! Расчетные параметры
        real(c_double):: v0
        real(c_double):: p_av_max
        real(c_double):: p_sn_max
        real(c_double):: p_kn_max
        real(c_double):: psi_sum
        real(c_double):: eta_k

        ! Статус ошибки (0-нет ошибки, 1-слишком много пороха, 2-превышено время выстрела)
        integer(c_int):: status

    end type ibproblem

    contains

    subroutine hello_from_fortran() bind(c, name="hello_from_fortran")
        print *, "hello from fortran"
    end subroutine hello_from_fortran

    subroutine set_problem(problem, p0, pv, ig_mass, t0) bind(c, name='set_problem')

        real(c_double), intent(in), value :: p0, pv, ig_mass, t0
        type(ibproblem), intent(inout) :: problem

        ! Исходные данные
        problem%p0 = p0
        problem%pv = pv
        problem%ig_mass = ig_mass
        problem%t0 = t0

        ! Расчетные значения
        problem%v0 = 0.0_dp
        problem%p_av_max = 1e5_dp
        problem%p_sn_max = 1e5_dp
        problem%p_kn_max = 1e5_dp
        problem%psi_sum = 0.0_dp
        problem%eta_k = 0.0_dp

        problem%status = 0

    end subroutine set_problem

    subroutine set_barrel(prob, d, q, s, w0, l_d, l_k, l0, kf) bind(c, name='set_barrel')

        type(ibproblem), intent(inout) :: prob
        real(c_double), intent(in), value :: d, q, s, w0, l_d, l_k, l0, kf

        type(barrel) :: barl

        barl = barrel(d, q, s, w0, l_d, l_k, l0, kf)

        prob%artsystem = barl

    end subroutine set_barrel

    subroutine add_powder(n_powd, powd_array, i, om, ro, f_powd, ti, jk, alpha, theta, zk, kappa1, lambda1, mu1,&
        kappa2, lambda2, mu2, gamma_f, gamma_jk) bind(c, name='add_powder')

        integer(c_int), intent(in), value :: n_powd
        type(powder), dimension(:), intent(inout) :: powd_array(n_powd)
        integer(c_int), intent(in), value :: i
        real(c_double), intent(in), value :: om, ro, f_powd, ti, jk, alpha, theta, zk, kappa1, lambda1, mu1,&
        kappa2, lambda2, mu2, gamma_f, gamma_jk

        type(powder) :: powd

        powd = powder(om, ro, f_powd, ti, jk, alpha, theta, zk, kappa1, lambda1, mu1,&
        kappa2, lambda2, mu2, gamma_f, gamma_jk)

        powd_array(i) = powd

    end subroutine add_powder

    real(c_double) function get_v0(prob) result(v0) bind(c, name='get_v0')

        type(ibproblem), intent(in) :: prob

        v0 = prob%v0

    end function get_v0

    real(c_double) function get_p_av_max(prob) result(p_av_max) bind(c, name='get_p_av_max')

        type(ibproblem), intent(in) :: prob

        p_av_max = prob%p_av_max

    end function get_p_av_max

    real(c_double) function get_psi_sum(prob) result(psi_sum) bind(c, name='get_psi_sum')

        type(ibproblem), intent(in) :: prob

        psi_sum = prob%psi_sum

    end function get_psi_sum

    real(c_double) function get_eta_k(prob) result(eta_k) bind(c, name='get_eta_k')

        type(ibproblem), intent(in) :: prob

        eta_k = prob%eta_k

    end function get_eta_k

    integer(c_int) function get_status(prob) result(status) bind(c, name='get_status')

        type(ibproblem), intent(in) :: prob

        status = prob%status

    end function get_status

end module ibstructs